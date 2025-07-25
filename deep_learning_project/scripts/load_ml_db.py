import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml_data.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

data_list = [
    (0.0, 0.0, 0),  # Input: (0, 0) -> Output: 0
    (0.0, 1.0, 1),  # Input: (0, 1) -> Output: 1
    (1.0, 0.0, 1),  # Input: (1, 0) -> Output: 1
    (1.0, 1.0, 0)   # Input: (1, 1) -> Output: 0
]

cursor.execute("""
  CREATE TABLE IF NOT EXISTS training_data (
    id INTEGER PRIMARY KEY,
    feature1 REAL,
    feature2 REAL,
    label INTEGER
  )
""")
cursor.executemany("INSERT INTO training_data (feature1, feature2, label) VALUES (?, ?, ?)", data_list)
conn.commit()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

df = pd.read_sql_query("SELECT * FROM training_data;", conn)
print(df.head())

conn.close()