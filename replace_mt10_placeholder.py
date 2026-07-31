import re
import sys

def replace_placeholder(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_block = """<div className="absolute inset-0 flex flex-col p-2 gap-2">
                                        <div className="flex-1 min-h-0 relative">
                                            <img src="results/fig_learning_curves.png" alt="MT10 Learning Curves" className="absolute inset-0 w-full h-full object-contain rounded" />
                                        </div>
                                        <div className="flex-1 min-h-0 relative">
                                            <img src="results/fig_sr_heatmap.png" alt="MT10 Success Rate Heatmap" className="absolute inset-0 w-full h-full object-contain rounded" />
                                        </div>
                                    </div>"""
    
    # Regex pattern to match the old block
    pattern = re.compile(
        r'<div className="absolute inset-0 flex items-center justify-center flex-col text-slate-500 p-8 text-center">.*?<IconBrain className="w-12 h-12 mb-4 opacity-50" />.*?<p className="font-mono text-sm">预留图位：Meta-World MT10 训练曲线 / 任务热力图</p>.*?<p className="text-xs mt-2 text-slate-600">\(展示 Head vs Tail 的性能反转\)</p>.*?</div>',
        re.DOTALL
    )
    
    new_content, count = pattern.subn(new_block, content)
    
    if count > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully replaced placeholder in {filepath} ({count} occurrences)")
    else:
        print(f"Placeholder not found in {filepath}")

replace_placeholder('index.html')
replace_placeholder('qtail-mvp-presentation.html')
