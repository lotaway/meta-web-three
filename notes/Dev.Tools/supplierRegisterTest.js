const puppeteer = require("puppeteer");

const host = "http://192.168.44.229:10010"
    , registerPath = "/mobi/cn/supplier/register.html"
    , typeInNode = function (page) {
        return async function (nodeName, value) {
            const node = await page.$(`[name=${nodeName}]`);

            await node.type(value);

            return true;
        };
    }
;

(async () => {
    const browser = await puppeteer.launch({
            executablePath: "D:/Program Files/chromium/chrome.exe",
            headless: false
        })
        , page = await browser.newPage()
        , tin = typeInNode(page)
    ;

    await page.goto(host + registerPath);
    const timeStamp = Date.now()
        , pwd = "123123"
        , delay = 300
    ;
    await tin("user_name", "userName-" + timeStamp, {delay});
    await tin("supplierName", "供应商名称-" + timeStamp, {delay});
    await tin("password", pwd, {delay});
    await tin("passwordConfirm", pwd, {delay});
    await tin("email", timeStamp + "@qq.com", {delay});
    await tin("mobile", "15999948166", {delay});
    let btnMobileCode = await page.$(".btnMobileCode", {delay});
    btnMobileCode.click();
    await tin("supplierAddress", "企业地址-" + timeStamp, {delay});
    await tin("supplierUrl", "企业网址-" + timeStamp, {delay});
    await tin("supplierRemark", "介绍一下-" + timeStamp, {delay});
})();