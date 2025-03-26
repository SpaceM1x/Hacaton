import sqlite3

def init_db():
    conn = sqlite3.connect('components.db')
    c = conn.cursor()

    # Drop the existing table if it exists to recreate with the correct schema
    c.execute('DROP TABLE IF EXISTS components')

    # Create the table with the correct column names matching the INSERT statement
    c.execute('''CREATE TABLE components
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  model TEXT NOT NULL,
                  characteristics TEXT,
                  quantity INTEGER NOT NULL,
                  category TEXT)''')

    # Расширенные тестовые данные с более подробной информацией
    components = [
        ('Intel Core i5-13700K', 'Количество ядер: 16, Базовая частота: 3.4 ГГц, Кэш-память: 30 МБ', 15, 'Процессор'),
        ('ASUS ROG Strix Z790-E Gaming WiFi', 'Сокет: LGA 1700, Формфактор: ATX, Чипсет: Intel Z790', 8, 'Материнская плата'),
        ('Corsair Vengeance RGB PRO 32GB', 'Объем: 32 ГБ, Тип: DDR4, Частота: 3600 МГц', 20, 'Оперативная память'),
        ('Samsung 980 PRO', 'Объем: 1 ТБ, Интерфейс: PCIe 4.0 NVMe, Скорость чтения: до 7000 МБ/с', 12, 'Накопитель'),
        ('NVIDIA GeForce RTX 4090', 'Видеопамять: 24 ГБ GDDR6X, Базовая частота: 2.2 ГГц', 5, 'Видеокарта')
    ]

    c.executemany('''INSERT INTO components 
                     (name, model, characteristics, quantity, category) 
                     VALUES (?, ?, ?, ?, ?)''', components)

    conn.commit()
    conn.close()
    print("База данных компонентов инициализирована.")

if __name__ == '__main__':
    init_db()