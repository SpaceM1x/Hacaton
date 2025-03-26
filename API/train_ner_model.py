import ast
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, Features, Sequence, ClassLabel, Value


def load_training_data_from_file(filename='lEARNING.txt'):
    """
    Загружает training_data из указанного файла
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            # Находим определение training_data
            training_data_str = content.split('training_data = ')[1]

            # Безопасный способ парсинга питоновского листа с помощью ast
            training_data = ast.literal_eval(training_data_str)

            return training_data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return []


def train_ner_model(
        filename='lEARNING.txt',
        model_name="Davlan/bert-base-multilingual-cased-ner-hrl",
        output_dir="./custom_ner_model"
):
    # Загрузка данных из файла
    training_data = load_training_data_from_file(filename)

    if not training_data:
        print("Не удалось загрузить данные для обучения.")
        return

    # Подготовка labels
    def prepare_labels(data):
        # Создаем метки с префиксами B- и I-
        labels = set()
        for data_item in data:
            for label in data_item['entities']:
                labels.add(f'B-{label["label"]}')
                labels.add(f'I-{label["label"]}')

        # Добавляем метку O для outside (не часть сущности)
        labels.add('O')

        label2id = {label: idx for idx, label in enumerate(sorted(labels))}
        id2label = {idx: label for label, idx in label2id.items()}
        return label2id, id2label

    # Токенизация и аннотирование данных
    def tokenize_and_align_labels(example, tokenizer, label2id):
        text = example['text']
        entities = example['entities']

        # Токенизация текста
        tokenized = tokenizer(text, truncation=True, is_split_into_words=False)

        # Инициализация меток
        labels = ['O'] * len(tokenized.input_ids)

        for entity in entities:
            start, end, label = entity['start'], entity['end'], entity['label']
            entity_tokens = tokenizer(text[start:end], add_special_tokens=False)['input_ids']

            # Поиск токенов сущности в тексте
            text_tokens = tokenizer(text, add_special_tokens=False)['input_ids']
            for i in range(len(text_tokens) - len(entity_tokens) + 1):
                if text_tokens[i:i + len(entity_tokens)] == entity_tokens:
                    labels[i] = f'B-{label}'
                    labels[i + 1:i + len(entity_tokens)] = [f'I-{label}'] * (len(entity_tokens) - 1)
                    break

        tokenized['labels'] = labels
        return tokenized

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Подготовка label mapping
    label2id, id2label = prepare_labels(training_data)

    # Загрузка модели с новым количеством меток
    config = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
    ).config

    config.id2label = id2label
    config.label2id = label2id

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config
    )

    # Создание Dataset
    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        remove_columns=dataset.column_names
    )

    # Разделение на train и validation
    dataset = dataset.train_test_split(test_size=0.2)

    # Аргументы тренировки
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    # Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    # Обучение модели
    trainer.train()

    # Сохранение модели и токенизатора
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Модель обучена и сохранена в {output_dir}")
    print(f"Обнаруженные метки: {list(label2id.keys())}")


def main():
    train_ner_model()


if __name__ == "__main__":
    main()