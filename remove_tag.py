import re

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Target the div
    pattern = r'<div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-qpurple/30 bg-qpurple/10 text-qpurple text-xs font-mono mb-8">\s*<span className="w-2 h-2 rounded-full bg-qpurple animate-pulse"></span>\s*Quafu 量子\+AI赛道 申报项目\s*</div>'
    
    new_content = re.sub(pattern, '', content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
print("Div removed!")
