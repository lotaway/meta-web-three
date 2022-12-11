const puppeteer = require("puppeteer")
    , path = require('path')
;

let webPath = "http://192.168.44.230:10018/Admin/Index";

(async () => {
    const browser = await puppeteer.launch({
            headless: false
        })
        , page = await browser.newPage()
    ;

    await page.goto(webPath);
    let node = await page.$("#txt_Password");
    await node.click();

})();