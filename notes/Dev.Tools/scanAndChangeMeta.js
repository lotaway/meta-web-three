//  todo 扫描所有模板的meta信息并与公共文件对比，剔除相同部分并引入公共文件
const fs = require("fs")
    , path = require('path')
    , projectRootPath = require("./projectRootPath").shopbest.root
;

/*
fs.readDir("/templates/mobi/Car", function (err, filePaths) {
    if (err)
        throw new Error("读取错误：" + err);
    else
        filePaths.forEach(function (filePath) {
            fs.readFile(filePath, function (err, file) {

            })
        });
});*/


let tabContentClass = "tabContent"
    , filePath = path.join(projectRootPath, "Micronet.Mvc/templates/backup/mobi/index.html")
;

fs.readFile(filePath, {
    encoding: "utf8"
}, function (err, data) {
    if (err) {
        return console.log("读取错误：" + err);
    }
    let rule = new RegExp(`(<ul class="list) ${tabContentClass}" data-id="([^"]+)(">(?:(?!</ul>).)*</ul>)`, "g")
        , matches = data.match(rule);
    if (matches)
        matches.forEach(function (item) {
            console.log("<!--开始-->" + item + "<!--结束-->");
        });
    else
        console.log("no match");
    // const newData = data.replace(rule, `<div class="${tabContentClass}" data-id="$2">$1$3<a class="btn-more">更多＞＞</a></div>`);
    /*fs.writeFile(filePath, function (err, result) {
        if (err) {
            return console.log("写入错误：" + err);
        }
        return console.log("替换成功：" + result);
    });*/
});