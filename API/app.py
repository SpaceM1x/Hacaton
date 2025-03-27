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


# Улучшенная функция извлечения компонентов
def extract_components(text):
    # Расширенные списки ключевых слов и производителей
    component_keywords = {
        'motherboard': ['материнская плата', 'мат. плата', 'mainboard', 'motherboard', 'плата'],
        'cpu': ['процессор', 'цпу', 'cpu', 'central processing unit', 'процессор'],
        'ram': ['оперативная память', 'ram', 'озу', 'память', 'dimm', 'ddr', 'memory'],
        'gpu': ['видеокарта', 'gpu', 'graphics card', 'графический процессор', 'видеокарта'],
        'ssd': ['ssd', 'жесткий диск', 'hdd', 'накопитель', 'storage', 'nvme', 'pcie', 'жд']
    }

    manufacturers = [
        'intel', 'amd', 'nvidia', 'asus', 'msi', 'gigabyte', 'samsung',
        'corsair', 'crucial', 'western digital', 'seagate', 'asrock',
        'palit', 'sapphire', 'zotac', 'powercolor', 'evga',
        'g.skill', 'kingston', 'team', 'hp', 'dell', 'rox', 'atx'
    ]

    # Улучшенные регулярные выражения для извлечения полных компонентов
    component_patterns = {
        'motherboard': [
            r'(?i)(материнская\s*плата|motherboard)(?:\s*модель)?[:]\s*([^\n]+)',
            r'(?i)(asus|msi|gigabyte|asrock)\s*([^\n]+\s*(?:z\d+|b\d+|x\d+|gaming|pro)[^\n]*)',
        ],
        'cpu': [
            r'(?i)(процессор|cpu)(?:\s*модель)?[:]\s*([^\n]+)',
            r'(?i)(intel|amd)\s*(core\s*i\d+[^\n]+|ryzen[^\n]+)',
        ],
        'ram': [
            r'(?i)(оперативная\s*память|ram|озу)(?:\s*модель)?[:]\s*([^\n]+)',
            r'(?i)(corsair|g\.skill|kingston|crucial)\s*([^\n]+\s*(?:\d+\s*(?:гб|gb)|ddr\d+)[^\n]*)',
        ],
        'gpu': [
            r'(?i)(видеокарта|gpu|graphics\s*card)(?:\s*модель)?[:]\s*([^\n]+)',
            r'(?i)(nvidia|amd)\s*(geforce|rtx|radeon[^\n]+)',
        ],
        'ssd': [
            r'(?i)(ssd|накопитель|жесткий\s*диск)(?:\s*модель)?[:]\s*([^\n]+)',
            r'(?i)(samsung|crucial|western\s*digital|seagate)\s*([^\n]+\s*(?:\d+\s*(?:тб|гб|tb|gb)|nvme)[^\n]*)',
        ]
    }


    # Функция для поиска компонентов по типу
    def find_components(text, patterns, component_type):
        components = []
        # Первый проход - поиск полных совпадений
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                # Если match - кортеж, берем второй элемент (имя компонента)
                component = match[1] if isinstance(match, tuple) else match
                if component and len(component.strip()) > 3:
                    # Очистка и обработка имени компонента
                    cleaned_component = re.sub(r'\s+', ' ', component.strip())

                    # Проверка на дубликаты
                    if not any(existing['name'] == cleaned_component for existing in components):
                        components.append({
                            'type': component_type,
                            'name': cleaned_component,
                            'characteristics': extract_characteristics(text, cleaned_component)
                        })

        return components

    # Извлечение компонентов по каждому типу
    extracted_components = []
    for component_type, patterns in component_patterns.items():
        extracted_components.extend(find_components(text, patterns, component_type))

    return extracted_components


# Извлечение характеристик с более контекстным подходом
def extract_characteristics(text, component_name):
    try:
        # Контекстный поиск характеристик
        context_search = re.search(
            fr"{re.escape(component_name)}.*?(\n.*?)+?(?=\n\d|\n[А-Я]|\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if context_search:
            context = context_search.group(0)
            return context.strip()

        return "Характеристики не найдены"
    except Exception as e:
        return f"Ошибка извлечения: {str(e)}"


# Сопоставление извлеченных компонентов с базой данных
def match_components(extracted, db_components):
    matched = []
    for item in extracted:
        best_match = None
        best_score = 0

        for db_item in db_components:
            # Более сложное сопоставление с несколькими критериями
            model_similarity = calculate_match_score(item['name'], db_item[1])

            # Дополнительные критерии сопоставления
            extra_criteria_score = 0
            if item.get('characteristics'):
                extra_criteria_score = calculate_match_score(
                    item['characteristics'],
                    db_item[2]
                )

            # Взвешенная оценка совпадения
            combined_score = (
                    model_similarity * 0.6 +
                    extra_criteria_score * 0.4
            )

            if combined_score > best_score:
                best_score = combined_score
                best_match = {
                    'required': item,
                    'available': (db_item[1], db_item[3]),
                    'match_score': combined_score
                }

        # Снижаем порог совпадения до 25%
        if best_score >= 25:
            matched.append(best_match)

    return matched


# Вычисление степени совпадения между текстами
def calculate_match_score(req_text, db_text):
    try:
        # Предобработка текста
        req_text = req_text.lower()
        db_text = db_text.lower()

        # Удаление специальных символов и лишних пробелов
        req_text = re.sub(r'[^\w\s]', '', req_text)
        db_text = re.sub(r'[^\w\s]', '', db_text)

        # Кодирование текстов
        embeddings = comparison_model.encode([req_text, db_text])

        # Вычисление косинусного сходства
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        # Преобразование в процентную шкалу с более мягкой шкалой
        return round(similarity * 100, 1)
    except Exception as e:
        print(f"Ошибка при вычислении сходства: {e}")
        return 0


# Генерация спецификации: объединение извлечения и сопоставления
def generate_specification(tech_spec_text, db_components):
    # Извлечение компонентов
    extracted_components = extract_components(tech_spec_text)

    print("Извлеченные компоненты:")
    for component in extracted_components:
        print(f"- {component['name']}: {component.get('characteristics', 'Без характеристик')}")

    print("\nКомпоненты в базе данных:")
    for db_component in db_components:
        print(f"- {db_component[0]} ({db_component[1]})")

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