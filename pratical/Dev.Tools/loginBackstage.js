//  登录后台
class LoginBackstage {

    constructor(host) {
        this.host = host;
    }

    async login({account, password, browser}) {
        const page = await browser.newPage();

        await page.goto(this.host + "/backstage/login.aspx");

        let userNameWriteFinish = page.$("#txtUserName").then(ele => ele.type(account))
            , passwordWriteFinish = userNameWriteFinish.then(page.$("#txtPassword").then(ele => ele.type(password)))
        ;


        return new Promise((resolve, reject) => {
            page.on("response", response => {
                response.url.indexOf("backstage/index.aspx") > -1 ? resolve(page) : "";
            });
            // Promise.all([passwordWriteFinish]).then(page.$("#btnSubmit").then(ele => ele.click()));
            // await page.waitForXPath("/backstage/index.aspx");
        });
    }

}

module.exports = LoginBackstage;