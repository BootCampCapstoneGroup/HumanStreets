import os

def clean_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        
        if b'\x00' in content:
            print(f"fixing {filepath} (detected null bytes/UTF-16)")
            # Try decoding as utf-16 (PowerShell default)
            try:
                text = content.decode('utf-16')
            except:
                # Fallback: try utf-16-le
                try:
                    text = content.decode('utf-16-le')
                except:
                    # Fallback: just strip nulls (risky but often works for ASCII with null padding)
                    text = content.replace(b'\x00', b'').decode('utf-8', errors='ignore')
            
            # Write back as UTF-8
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text.strip())
        else:
            print(f"clean {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Scanning {root_dir}...")
    
    for dirpath, _, filenames in os.walk(root_dir):
        if "__pycache__" in dirpath:
            continue
        for filename in filenames:
            if filename.endswith(".py"):
                clean_file(os.path.join(dirpath, filename))

if __name__ == "__main__":
    main()
