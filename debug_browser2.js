const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({headless: "new"});
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('BROWSER_LOG:', msg.text()));
  page.on('pageerror', error => console.log('BROWSER_ERROR:', error.message));

  await page.goto('http://localhost:6225/', { waitUntil: 'networkidle0' });
  const rootHTML = await page.evaluate(() => document.getElementById('root').innerHTML);
  console.log("Root length:", rootHTML.length);
  if (rootHTML.length === 0) {
    console.log("Root is empty");
  }
  await browser.close();
  process.exit(0);
})();
