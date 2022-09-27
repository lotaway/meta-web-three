/**
 * todo
 * 1、通过几个预设好的影视网址以及结构，打开时在每天拉取一次网站N页内的数据直到日期相同或达到配置预设的上限页数，合并相同名称不同季度不同来源的剧，保留资源名称和链接，并存储下来，显示时按更新时间排序；
 * 2、初始化将按配置拉取N页以内的剧，细节按步骤1处理；
 * 3、可以订阅放送中、季度完结、未开播的剧，当有更新时优先显示；
 * 4、增加点击下载缓存功能？
 */
const request = require("request")
    , cheerio = require("cheerio")
    , {pageLoad} = require("../nodejs/utils/loader")
;
let keyword = ""    //  搜索关键字
    , urlArr = [
        {
            listUrl: "https://m.dadatu5.com/vodsearch.html?wd=" + keyword
        }
    ]
;

function start() {
    urlArr.forEach(function (item) {
        page("https://m.dadatu5.com/vodsearch.html?wd=" + keyword).then(body => {
            let result
                , $ = cheerio.load(body)    //  将文档内容解析成类JQuery对象
                , promise_array = []
            ;


        });
    });
}