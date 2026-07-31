import re

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Target the img tag and replace its src
    pattern = r'<img src="results/fig_quantum_prob_dist\.png"'
    replacement = r'<img src="data/Code_Generated_Image.png"'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
print("Image source updated!")
