import re

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the empty or broken logo div with the correct one wrapped in an anchor
    pattern = r'<div className="flex items-center gap-2">\s*<IconAtom className="text-qcyan" />\s*</div>'
    
    replacement = """<a href="#" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                        <IconAtom className="text-qcyan" />
                        <span className="font-bold text-lg tracking-wider text-white">From QRCS to Future Scales</span>
                    </a>"""
    
    content = re.sub(pattern, replacement, content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

print("Restored logo to homepage!")
