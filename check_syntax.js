const fs = require('fs');
const babel = require('@babel/core');

const content = fs.readFileSync('index.html', 'utf-8');
const scriptMatch = content.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);

if (scriptMatch) {
  const code = scriptMatch[1];
  try {
    babel.transformSync(code, {
      presets: ['@babel/preset-react'],
      filename: 'index.jsx'
    });
    console.log("Syntax is valid!");
  } catch (e) {
    console.error("Syntax Error:", e.message);
  }
} else {
  console.log("Could not find babel script");
}
