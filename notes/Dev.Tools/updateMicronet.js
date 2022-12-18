const puppeteer = require("puppeteer")
    , path = require('path')
    , shell = require('shelljs')
    , tesseract = require('node-tesseract')
;

let webPath = "http://www.micronet.com.cn/backstage/",
    login = {
        userName: "lw",
        password: "lw*()"
    };

(async () => {
    const browser = await puppeteer.launch({
            headless: false
        })
        , page = await browser.newPage()
    ;

    await page.goto(webPath);
    /*page.evaluate(async login => {
        let txtUserName = document.getElementById("txtUserName")
            , txtPassword = document.getElementById("txtPassword")
            , txtCheckCode = document.getElementById("txtCheckCode")
        ;

        txtUserName.click();
        txtUserName.value = login.userName;
        txtPassword.click();
        txtPassword.value = login.password;
        txtCheckCode.click();

        await browser.close();
    }, login);*/

    let txtUserName = await page.$("#txtUserName");
    await txtUserName.click();
    await txtUserName.type(login.userName);
    let txtPassword = await page.$("#txtPassword");
    await txtPassword.click();
    await txtPassword.type(login.password);
    let checkcode = await page.$("#checkcode");
    let codeImg = checkcode.src;
    // http://blog.csdn.net/neal1991/article/details/51249823 验证码识别
    //  https://www.cnblogs.com/jianqingwang/p/6978724.html  tesseract-ocr 图像识别
    let txtCheckCode = await page.$("#txtCheckCode");
    let loginContent = await page.$(".loginContent");
    await page.screenshot({
        path: path.join(__dirname, "./micronet.jpeg"),
        type: "jpeg",
        quality: 100,
        clip: {
            x: loginContent.offsetLeft + txtCheckCode.offsetLeft,
            y: loginContent.offsetTop + txtCheckCode.offsetTop,
            width: txtCheckCode.offsetWidth || txtCheckCode.width || 100,
            height: txtCheckCode.offsetHeight || txtCheckCode.height || 50
        }
    });
    // await shell.exec("convert 1.jpg -colorspace gray -normalize -threshold 50% 1.tif");
    // await shell.exec("tesseract 1.tif result");
    // await txtCheckCode.type("");
    // await page.press("Enter");
    await browser.close();

})();