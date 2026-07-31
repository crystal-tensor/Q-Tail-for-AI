import re

pattern = r'<p className="text-xl md:text-2xl text-slate-400 font-light tracking-wide mb-8">\s*Quantum-Guided Tail Distribution Engine for Embodied Learning\s*</p>'

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = re.sub(pattern, '', content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
print("Paragraph removed!")
