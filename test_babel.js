const fs = require('fs');
const babel = require('@babel/core');

const html = fs.readFileSync('index.html', 'utf-8');
const scriptMatch = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);

if (scriptMatch) {
    const code = scriptMatch[1];
    try {
        babel.transformSync(code, {
            presets: ['@babel/preset-react']
        });
        console.log('Babel parsing successful!');
    } catch (e) {
        console.error('Babel parsing failed:', e);
        process.exit(1);
    }
} else {
    console.error('No babel script found');
    process.exit(1);
}
