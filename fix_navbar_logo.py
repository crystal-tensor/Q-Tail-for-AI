import re

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Target only the Navbar logo area
    pattern = r'(<a href="#" className="flex items-center gap-2 hover:opacity-80 transition-opacity">\s*)<IconAtom className="text-qcyan" />(\s*)<span className="font-bold text-lg tracking-wider text-white">(.*?)</span>(\s*</a>)'
    
    # Shrink svg (w-5 h-5), hide span and minimize it (hidden w-0 h-0 overflow-hidden)
    replacement = r'\1<IconAtom className="text-qcyan w-5 h-5" />\2<span className="hidden w-0 h-0 overflow-hidden">\3</span>\4'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
print("Navbar logo updated!")
