import re

with open('qtail-mvp-presentation.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the position of DataEngineArchitecture
content = re.sub(r'const QuantumAILoop = \(\) => \([\s\S]*?        \);\n\n', '', content)

with open('add_quantum_ai_loop.py', 'r', encoding='utf-8') as f:
    updater_content = f.read()

new_component_start = updater_content.find('new_component = """\n') + len('new_component = """\n')
new_component_end = updater_content.find('\n"""', new_component_start)
new_component = updater_content[new_component_start:new_component_end]

content = content.replace("        const DataEngineArchitecture", new_component + "\n\n        const DataEngineArchitecture")

with open('qtail-mvp-presentation.html', 'w', encoding='utf-8') as f:
    f.write(content)
print("Fixed duplicates in presentation")