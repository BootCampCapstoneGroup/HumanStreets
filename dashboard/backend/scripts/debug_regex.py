import re

def test_strip(query):
    print(f"Original: {query}")
    print(f"Hex: {[hex(ord(c)) for c in query]}")
    
    # Logic from sql_agent.py
    is_match = 'حي' in query
    print(f"'حي' in query: {is_match}")
    
    if is_match:
        # Regex from sql_agent.py
        # Match % followed by optional space, then حي, then REQUIRED space
        query_fixed = re.sub(r"%\s*حي\s+", "%", query)
        query_fixed = re.sub(r"'\s*حي\s+", "'", query_fixed)
        print(f"Fixed:    {query_fixed}")
    else:
        print("No change")
    print("-" * 20)

# Test cases
test_strip("WHERE name ILIKE '%حي النرجس%'")
test_strip("WHERE name ILIKE '% حي النرجس%'")
test_strip("WHERE name ILIKE '%حي  النرجس%'")
test_strip("WHERE name ILIKE '%حى النرجس%'") # Alef Maksura variation?

# Common typo: Alef Maksura instead of Ya
hayy_typo = "حى" # \u062d\u0649
test_strip(f"WHERE name ILIKE '%{hayy_typo} النرجس%'")
