import sys
sys.stdout.reconfigure(encoding='utf-8')

from sqlalchemy import create_engine, text
from app.core.config import settings

e = create_engine(settings.DATABASE_URL)
conn = e.connect()

# Find neighborhoods with narjis/malqa
res = conn.execute(text("SELECT name FROM neighborhoods WHERE name ILIKE '%نرجس%' OR name ILIKE '%ملقا%'"))
print("Found neighborhoods matching Narjis or Malqa:")
for r in res.fetchall():
    print(f"  - {r[0]}")

# Check how many total neighborhoods
res = conn.execute(text("SELECT COUNT(*) FROM neighborhoods"))
print(f"\nTotal neighborhoods: {res.fetchone()[0]}")

# List all names
res = conn.execute(text("SELECT name FROM neighborhoods ORDER BY name"))
print("\nAll neighborhood names:")
for r in res.fetchall():
    print(f"  - {r[0]}")
