import ftfy
import os
import fnmatch

def fix_mojibake_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    fixed_text = ftfy.fix_text(original_text)
    
    if fixed_text != original_text:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_text)
        print(f"Fixed: {filepath}")
    else:
        print(f"No changes needed: {filepath}")

def main():
    exclude_dirs = {'.git', '.venv'}
    patterns = ['*.py', '*.bak']
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                filepath = os.path.join(root, filename)
                fix_mojibake_in_file(filepath)

if __name__ == '__main__':
    main()