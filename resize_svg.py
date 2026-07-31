import re

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Target the specific anchor tag
    pattern = r'<a href="#advantages" className="hover:text-qcyan transition-colors flex items-center gap-1.5"><IconAtom className="text-qcyan w-5 h-5" />核心优势</a>'
    
    # Increase gap from 1.5 to 2.5, and increase icon size from w-5 h-5 to w-6 h-6
    replacement = r'<a href="#advantages" className="hover:text-qcyan transition-colors flex items-center gap-2.5"><IconAtom className="text-qcyan w-[22px] h-[22px]" />核心优势</a>'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)

print("SVG resized and gap increased!")