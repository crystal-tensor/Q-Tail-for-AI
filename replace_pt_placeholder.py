import re

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Target the placeholder div
    pattern = r'<div className="absolute inset-0 flex items-center justify-center flex-col text-slate-500">\s*<IconChart className="w-12 h-12 mb-4 opacity-50" />\s*<p className="font-mono text-sm">预留图位：PT分布拟合图 / 概率分布统计图</p>\s*<p className="text-xs mt-2 text-slate-600">\(将在最终汇报时填入真实 matplotlib 出图\)</p>\s*</div>'
    
    # Replace with the image tag
    replacement = r'<div className="absolute inset-0 flex items-center justify-center bg-black/40 p-2"><img src="results/fig_quantum_prob_dist.png" alt="PT Distribution" className="w-full h-full object-contain opacity-90 hover:opacity-100 transition-opacity rounded" /></div>'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
print("PT placeholder replaced!")