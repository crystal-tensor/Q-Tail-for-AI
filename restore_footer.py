import re

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the empty or broken logo div in the footer
    pattern_footer = r'<div className="flex items-center justify-center gap-2 mb-6">\s*<IconAtom className="text-qcyan w-6 h-6" />\s*</div>'
    
    replacement_footer = """<div className="flex items-center justify-center gap-2 mb-6">
                        <IconAtom className="text-qcyan w-6 h-6" />
                        <span className="font-bold text-xl tracking-wider text-white">From QRCS to Future Scales</span>
                    </div>"""
    
    content = re.sub(pattern_footer, replacement_footer, content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

print("Restored logo to footer!")
