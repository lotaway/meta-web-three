const puppeteer = require("puppeteer")
    , path = require("path")
    , LoginBackstage = require("./loginBackstage")
;

const host = "http://192.168.44.229:10019"
    , frontUrl = `${host}/mobi/cn/blackCard_agreement`
    , backstage = {
        account: "admin",
        password: "123123",
        blackCard: {
            agreement: {
                title: "测试的标题",
                content: "测试的内容"
            }
        }
    }
;

(async () => {
    const browser = await puppeteer.launch({
        executablePath: "D:/Program Files/chromium/chrome.exe"
        , headless: false
    })
        , frontPage = await browser.newPage();

    let loginBackstageReady = await (new LoginBackstage(host)).login({
        account: backstage.account,
        password: backstage.password,
        browser
    });
    let mainFrame = await loginBackstageReady(async backstagePage => await backstagePage.$("#mainframe"));
    await mainFrame.$(".ke-edit-iframe").$("body").getProperty("innerHTML");
    await frontPage.goto(frontUrl);
    let screenShotReady = frontPage.screenshot({
        path: path.join(__dirname, "./blackCardAgreement.jpeg"),
        type: "jpeg",
        quality: 100
    });
    let container = await frontPage.$(".containerBlackCardAgreement");
    if (container.$(".title").getProperty("innerHTML") === backstage.blackCard.agreement.title && container.$(".content").getProperty("innerHTML") === backstage.blackCard.agreement.content) {
        console.log("内容相同");
    }
    else {
        console.log("内容有差异");
    }
    console.log("执行...");

    await Promise.all([screenShotReady]);
    await browser.close();
})();