import re

with open('qtail-mvp-presentation.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract the old Architecture component
match = re.search(r'const Architecture = \(\) => \([\s\S]*?\n        \);[\s]*const InteractiveMVPTarget', content)
if not match:
    print("Could not find Architecture component in presentation")
else:
    old_arch = match.group(0).replace('const InteractiveMVPTarget', '')
    
    with open('update_arch.py', 'r', encoding='utf-8') as f:
        updater_content = f.read()
    
    # Simple extraction of new_arch from the previous script
    new_arch_start = updater_content.find('new_arch = """') + len('new_arch = """')
    new_arch_end = updater_content.find('"""', new_arch_start)
    new_arch = updater_content[new_arch_start:new_arch_end]
    
    content = content.replace(old_arch, new_arch)
    content = content.replace("<Architecture />", "<DataEngineArchitecture />")
    
    with open('qtail-mvp-presentation.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Replaced successfully in presentation!")