from sqlalchemy import create_engine, text
import pandas as pd

DB_URL = "postgresql://postgres:12345@localhost:5432/capstone"
engine = create_engine(DB_URL)

print("--- Neighborhood Scores ---")
with engine.connect() as conn:
    df = pd.read_sql("SELECT name, avg_walkability FROM neighborhoods", conn)
    print(df.describe())
    print("\nSample (Head):")
    print(df.head())
    print("\nSample (High Scores > 50):")
    print(df[df['avg_walkability'] > 50])
    
    print("\n--- H3 Grid Scores ---")
    h3_df = pd.read_sql("SELECT avg_street_score FROM h3_grid", conn)
    print(h3_df.describe())
