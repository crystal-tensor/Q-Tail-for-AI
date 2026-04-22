const fs = require('fs');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;

const content = fs.readFileSync('index.html', 'utf-8');
const dom = new JSDOM(content, {
  runScripts: "dangerously",
  resources: "usable",
  url: "http://localhost/"
});

dom.window.document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    console.log("Root HTML exists:", !!dom.window.document.getElementById('root').innerHTML);
    if (dom.window.onerror) {
        console.log("Window Error:", dom.window.onerror);
    }
  }, 3000);
});
