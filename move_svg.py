import re

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Remove the SVG from the left-most logo area
    content = re.sub(r'<IconAtom className="text-qcyan w-5 h-5" />\s*<span className="hidden w-0 h-0 overflow-hidden">', '<span className="hidden w-0 h-0 overflow-hidden">', content)
    
    # 2. Add the SVG to the left of the '核心优势' text in the Navbar
    pattern_advantages = r'<a href="#advantages" className="hover:text-qcyan transition-colors">核心优势</a>'
    replacement_advantages = r'<a href="#advantages" className="hover:text-qcyan transition-colors flex items-center gap-1.5"><IconAtom className="text-qcyan w-5 h-5" />核心优势</a>'
    
    content = re.sub(pattern_advantages, replacement_advantages, content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

print("SVG moved to the '核心优势' link!")