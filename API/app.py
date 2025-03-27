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
    model="DeepPavlov/rubert-base-cased",
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
    # Улучшенные паттерны для извлечения оперативной памяти
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
            # Новые, более специфичные паттерны для RAM
            r'(?i)(оперативная\s*память|ram|озу)(?:\s*модель)?[:]\s*([^\n]+)',
            r'(?i)(corsair|g\.skill|kingston|crucial)\s*([^\n]+(?:\d+\s*(?:гб|gb)|ddr\d+)[^\n]*)',
            # Добавляем извлечение с детальными характеристиками
            r'(?i)(ddr\d+)\s*([^\n]+(?:\d+\s*(?:гб|gb)|pro|vengeance)[^\n]*)',
            r'(?i)(vengeance\s*rgb\s*pro)\s*(\d+\s*(?:гб|gb))[^\n]*',
        ],
        'gpu': [
            r'(?i)(видеокарта|gpu|graphics\s*card)(?:\s*модель)?[:]\s*([^\n]+)',
            r'(?i)(nvidia|amd|geforce|rtx|radeon)\s*([^\n]+(?:rtx\s*\d+|geforce|radeon)[^\n]*)',
        ],
        'ssd': [
            r'(?i)(ssd|накопитель|жесткий\s*диск|storage)(?:\s*модель)?[:]\s*([^\n]+)',
            r'(?i)(samsung|crucial|western\s*digital|seagate|wd|sx\d+)\s*([^\n]+\s*(?:\d+\s*(?:тб|гб|tb|gb)|nvme|pro|evo)[^\n]*)',
        ]
    }

    # Функция для поиска компонентов по типу
    def find_components(text, patterns, component_type):
        components = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                # Ensure we get the component name
                component = match[1] if isinstance(match, tuple) else match
                if component and len(component.strip()) > 3:
                    # Clean and process component name
                    cleaned_component = re.sub(r'\s+', ' ', component.strip())

                    # Avoid duplicates
                    if not any(existing['name'] == cleaned_component for existing in components):
                        # Enhanced component extraction with more context
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
            # More sophisticated matching with multiple criteria
            # Специальная логика для RAM с учетом объема и типа памяти
            if item['type'] == 'ram':
                # Более точное извлечение объема и типа памяти
                volume_match = re.search(r'(\d+)\s*(?:гб|gb)', item['name'], re.IGNORECASE)
                ddr_match = re.search(r'(ddr\d+)', item['name'], re.IGNORECASE)
                volume_match_db = re.search(r'Объем:\s*(\d+)', db_item[2], re.IGNORECASE)
                ddr_match_db = re.search(r'Тип:\s*(ddr\d+)', db_item[2], re.IGNORECASE)

                volume = int(volume_match.group(1)) if volume_match else 0
                volume_db = int(volume_match_db.group(1)) if volume_match_db else 0

                ddr_type = ddr_match.group(1).lower() if ddr_match else ''
                ddr_type_db = ddr_match_db.group(1).lower() if ddr_match_db else ''

                # Проверка соответствия объема и типа памяти с более строгими условиями
                volume_score = 50 if volume > 0 and volume == volume_db else 0
                ddr_score = 30 if ddr_type and ddr_type == ddr_type_db else 0

                # Базовое сходство названий
                name_similarity = calculate_match_score(item['name'], db_item[1])
                manufacturer_match = any(
                    manuf.lower() in item['name'].lower()
                    for manuf in ['corsair', 'g.skill', 'crucial', 'kingston', 'team']
                )

                # Комбинированный score для RAM с учетом производителя
                combined_score = (
                        name_similarity * 0.2 +  # Название
                        volume_score +  # Объем памяти
                        ddr_score +  # Тип DDR
                        (20 if manufacturer_match else 0)  # Бонус за производителя
                )
            else:
                # Для других компонентов - прежняя логика
                model_similarity = calculate_match_score(item['name'], db_item[1])

                # Additional matching criteria
                extra_criteria_score = 0
                if item.get('characteristics'):
                    extra_criteria_score = calculate_match_score(
                        item['characteristics'],
                        db_item[2]
                    )

                # Weighted match score
                combined_score = (
                        model_similarity * 0.7 +
                        extra_criteria_score * 0.3
                )

            # Debug print to understand matching process
            print(f"Matching {item['name']} with {db_item[1]}: Score = {combined_score}")

            if combined_score > best_score:
                best_score = combined_score
                best_match = {
                    'required': item,
                    'available': (db_item[1], db_item[3]),  # Model and quantity
                    'match_score': combined_score
                }

        # Повышаем порог до 40%, чтобы избежать ложных срабатываний
        if best_score >= 40:
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