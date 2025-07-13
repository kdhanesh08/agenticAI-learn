import re

with open("my_notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

cleaned = re.sub(r"\s+", " ", text).strip()

with open("cleaned_notes.txt", "w", encoding="utf-8") as f:
    f.write(cleaned)

print("✅ Cleaned and saved as 'cleaned_notes.txt'")