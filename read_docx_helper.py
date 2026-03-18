import zipfile
import re
import sys
import os

def read_docx(file_path):
    try:
        with zipfile.ZipFile(file_path) as docx:
            content = docx.read('word/document.xml').decode('utf-8')
            # XML tags removal
            text = re.sub('<[^<]+?>', '', content)
            print(text)
    except Exception as e:
        print(f"Error reading docx: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_docx.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    if os.path.exists(filename):
        read_docx(filename)
    else:
        print(f"File not found: {filename}")
