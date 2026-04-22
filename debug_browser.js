const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch({headless: "new"});
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('BROWSER_LOG:', msg.text()));
  page.on('pageerror', error => console.log('BROWSER_ERROR:', error.message));

  await page.goto('http://localhost:6225/', { waitUntil: 'networkidle0' });
  const html = await page.content();
  if (html.includes('<div id="root"></div>')) {
    console.log("Root is empty!");
  } else {
    console.log("Root is NOT empty. Length:", html.length);
  }
  await browser.close();
  process.exit(0);
})();
