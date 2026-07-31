import re

with open('temp_results.js', 'r') as f:
    new_component = f.read()

for file_name in ['index.html', 'qtail-mvp-presentation.html']:
    with open(file_name, 'r') as f:
        content = f.read()
    
    # Find start and end of ResultsPlaceholder
    start_idx = content.find('const ResultsPlaceholder = () => ')
    end_idx = content.find('const Footer = () => ')
    
    if start_idx != -1 and end_idx != -1:
        new_content = content[:start_idx] + new_component + "        " + content[end_idx:]
        with open(file_name, 'w') as f:
            f.write(new_content)
        print(f"Replaced ResultsPlaceholder in {file_name}")
    else:
        print(f"Could not find markers in {file_name}")
