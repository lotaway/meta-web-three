const puppeteer = require("puppeteer")
    , path = require("path");

let webPath = "https://www.shopbest.com.cn";

//  示例
(async () => {
    //  启动浏览器
    const browser = await puppeteer.launch({
            executablePath: "D:/Program Files/chromium/chrome.exe", //  无法翻墙下载需要手动下载并指定chromium的位置
            // executablePath: "E:/workspace/project/nodeJsDemo/node_modules/puppeteer/.local-chromium/win64-515411/chrome-win32/chrome.exe",
            // headless: false //  不打开chromium
        })
        //  新建选项卡
        , page = await browser.newPage()
    ;
    //  跳转地址
    await page.goto(webPath);
    // 截图
    await page.screenshot({
        path: path.join(__dirname, "./", webPath.replace(new RegExp("[?/\\:*<>|\"]*", "g"), "") + ".png")
    });
    //  关闭页面或者选项卡
    await browser.close();
})();
/*

//  抓取接口请求的页面数据
const cheerio = require("cheerio")
    , axios = require("axios")
    , chalk = require("chalk")
    , mapLimit = require("async/mapLimit")
    , $util = require("./../helper/utils.js")
    , $config = require("..config.js")
    ;

$util.setConfig($config);
puppeteer.launch({
    handless: true
})
    .then(async browser => {
        let page = await browser.newPage();
        page.setViewport({
            width: 1024,
            height: 2048
        });
        page
            .waitForSelector("img")
            .then(async () => {
                executeCrowlPlan(browser, page)
            });
        page.on("requestfinished", result => {
            if (result.url.includes("clustrmaps.com")) {
                $util.onListenUrlChange(page)
            }
        });
        page.on("error", error => {
            console.log(chalk.red("whoops! there was an error"));
            console.log(error);
        })
    })*/
