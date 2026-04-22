const fs = require('fs');
const content = fs.readFileSync('index.html', 'utf-8');
const lines = content.split('\n');
console.log(lines.slice(lines.length - 20).join('\n'));
