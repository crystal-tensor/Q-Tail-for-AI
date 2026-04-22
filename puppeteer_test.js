const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({headless: "new"});
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('BROWSER_LOG:', msg.text()));
  page.on('pageerror', error => console.log('BROWSER_ERROR:', error.message));
  page.on('requestfailed', request => console.log('BROWSER_REQ_FAILED:', request.url(), request.failure().errorText));

  try {
    await page.goto('http://localhost:6225/');
    await new Promise(r => setTimeout(r, 1000));
  } catch(e) {
    console.log("PUPPETEER_ERROR", e);
  }
  await browser.close();
  process.exit(0);
})();
