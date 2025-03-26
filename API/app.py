from flask import Flask, render_template, request, redirect, url_for
from docx import Document
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import sqlite3
import os
import re

# Отключаем предупреждение о симлинках от Hugging Face
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Настройка Flask-приложения
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'docx'}  # Поддерживаем только .docx

# Создаем папку для загрузок, если она не существует
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Инициализация моделей
ner_model = pipeline(
    "ner",
    model="./custom_ner_model",  # Путь к обученной модели
    aggregation_strategy="simple"
)


comparison_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
gen_model_name = "UrukHan/t5-russian-summarization"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

# Проверка допустимого расширения файла
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Извлечение текста из .docx файла
def extract_text_from_docx(filepath):
    doc = Document(filepath)
    return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])

# Получение компонентов из базы данных с проверкой существования файла
def get_components_from_db():
    if not os.path.exists('components.db'):
        raise FileNotFoundError("База данных 'components.db' не найдена.")
    conn = sqlite3.connect('components.db')
    c = conn.cursor()
    c.execute(
        'SELECT name, model, characteristics, quantity, category FROM components WHERE quantity > 0 ORDER BY name')
    components = c.fetchall()
    conn.close()
    return components

# Расширенное извлечение компонентов с использованием NER и регулярных выражений
def extract_components(text):
    component_keywords = [
        # Расширенный список компонентов
        'процессор', 'цпу', 'cpu', 'central processing unit',
        'материнская плата', 'мат. плата', 'mainboard', 'motherboard',
        'оперативная память', 'ram', 'озу', 'memory',
        'ssd', 'жесткий диск', 'hdd', 'накопитель',
        'видеокарта', 'gpu', 'graphics card', 'графический процессор',
        'блок питания', 'psu', 'источник питания',
        'корпус', 'case', 'системный блок',
        'охлаждение', 'кулер', 'система охлаждения'
    ]

    manufacturers = [
        # Расширенный список производителей
        'intel', 'amd', 'nvidia', 'asus', 'msi', 'gigabyte', 'samsung',
        'corsair', 'crucial', 'western digital', 'seagate', 'asrock',
        'palit', 'sapphire', 'zotac', 'powercolor', 'evga'
    ]

    # Регулярные выражения для поиска компонентов
    component_patterns = [
        r'(?i)({keywords})\s+([\w\-\s]+\d+\w*)'.format(keywords='|'.join(component_keywords)),
        r'(?i)({manufacturers})\s+([\w\-\s]+\d+\w*)'.format(manufacturers='|'.join(manufacturers))
    ]

    # Извлечение компонентов через NER
    ner_results = ner_model(text)

    # Извлечение компонентов через регулярные выражения
    regex_components = []
    for pattern in component_patterns:
        regex_components.extend(re.findall(pattern, text, re.IGNORECASE))

    # Объединение результатов
    components = []
    seen = set()

    # Добавление результатов NER
    for entity in ner_results:
        if entity['entity_group'] in ['PRODUCT', 'MODEL']:
            component = entity['word']
            if component.lower() not in seen:
                components.append({
                    'name': component,
                    'characteristics': extract_characteristics(text, component)
                })
                seen.add(component.lower())

    # Добавление результатов regex
    for match in regex_components:
        component = match[1] if isinstance(match, tuple) else match
        if component.lower() not in seen:
            components.append({
                'name': component,
                'characteristics': extract_characteristics(text, component)
            })
            seen.add(component.lower())

    return components

# Извлечение характеристик компонента с помощью T5
def extract_characteristics(text, component_name):
    try:
        prompt = f"""
        Извлеки технические характеристики для {component_name} из текста:
        {text[:2000]}
        Ответ приведи в формате: Характеристика: Значение
        """
        inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = gen_model.generate(**inputs, max_length=400)
        return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Не удалось извлечь характеристики: {str(e)}"

# Сопоставление извлеченных компонентов с базой данных
def match_components(extracted, db_components):
    matched = []
    for item in extracted:
        for db_item in db_components:
            # Расширенное сопоставление с учетом модели, названия и характеристик
            similarity_score = calculate_match_score(
                f"{item['name']} {item.get('characteristics', '')}",
                f"{db_item[0]} {db_item[1]} {db_item[2]}"
            )

            if similarity_score >= 50:  # Порог совпадения
                matched.append({
                    'required': item,
                    'available': (db_item[1], db_item[3]),  # Модель и количество
                    'match_score': similarity_score
                })

    return matched

# Вычисление степени совпадения между компонентами
def calculate_match_score(req_text, db_text):
    try:
        # Нормализация текста
        req_text = req_text.lower()
        db_text = db_text.lower()

        # Кодирование текстов
        embeddings = comparison_model.encode([req_text, db_text])

        # Вычисление косинусного сходства
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        # Преобразование в процентную шкалу
        return round(similarity * 100, 1)
    except Exception as e:
        print(f"Ошибка при вычислении сходства: {e}")
        return 0

# Генерация спецификации: объединение извлечения и сопоставления
def generate_specification(tech_spec_text, db_components):
    # Извлечение компонентов
    extracted_components = extract_components(tech_spec_text)

    # Сопоставление с базой данных
    matched = match_components(extracted_components, db_components)

    return extracted_components, matched

# Основной маршрут приложения
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Извлечение текста
                text = extract_text_from_docx(filepath)

                # Получение компонентов из базы данных
                db_components = get_components_from_db()

                # Генерация спецификации
                extracted, matched = generate_specification(text, db_components)

                return render_template(
                    'display.html',
                    text=text,
                    extracted=extracted,
                    matched=matched,
                    db_components=db_components
                )

            except Exception as e:
                return f'Ошибка: {str(e)}'

    return render_template('index.html')

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True)