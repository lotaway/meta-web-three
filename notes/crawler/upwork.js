/**
 * 网络爬虫，简单获取页面内容 2016/5/8.
 */
const puppeteer = require('puppeteer')
const https = require("https")
const {join} = require("path")
const {createWriteStream} = require("fs")
const cheerio = require('cheerio')
const targetUrl = 'https://www.upwork.com/nx/find-work/'

function htmlStrToJson(htmlStr) {
    const $ = cheerio.load(htmlStr)
    let jsonArr = []
    console.log($.html())
    const jobList = $("section.up-card-list-section")
    jobList.each(index => {
        const item = $(this)
        const titleNode = item.find(".job-tile-title")
        const link = `${targetUrl}${titleNode.children("a").attr("href")}`
        const title = titleNode.text()
        const desc = item.find(".job-description-text").text()
        const jobType = item.find('data-test="job-type"').text()
        const budget = item.find('[data-itemprop="baseSalary"]').text()
        jsonArr.push({
            title,
            desc,
            link,
            jobType,
            budget
        })
    })
    return jsonArr
}

function saveData(dataJson) {
    console.log(dataJson)
    const writeStream = createWriteStream(join(__dirname, "./download.json"))
    writeStream.write(JSON.stringify(dataJson), (err) => {
        if (err) throw err
        console.log("done")
    })
    writeStream.close()
}

async function testPage() {
    const browser = await puppeteer.launch({
        headless: true
    })
    const page = await browser.newPage()
    await page.goto(targetUrl)
    const jobList = await page.$("section.up-card-list-section")
    console.log(await page.content())
    // jobList.$
    await browser.close()
}

testPage().then(() => {
    console.log("done")
})

/*
https
    .get(targetUrl, (res) => {
        let htmlStr = "";
        res.on('data', chunk => htmlStr += chunk)
        res.on('end', () => {
            saveData(htmlStrToJson(htmlStr))
        })
        res.on("error", err => {
            console.error(err)
        })
    })
    .on('error', () => console.log('catch error'))*/
