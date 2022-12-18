/**
 * 汕头招聘网爬取职位要求分布，TODO 控制并发请求数
 */
const cheerio = require("cheerio")
    , mongoose = require("mongoose")
    , db = mongoose.connect("mongodb://127.0.0.1:27017/job")
    , {pageLoad, LimitLoad} = require("../nodejs/utils/loader")
    , host = "http://www.stzp.cn"
    , searchPath = "/search/"
;
let {StJobModel} = require("./schema/stJobKeyWord")
    , keyword = "前端"
    , listPageUrlArr = ["offer_search_result.aspx?jtype1Hidden=&jcity1Hidden=101700&keyword=" + keyword]  //  分页链接
    , reqPromise = Promise.resolve()

;

StJobModel.find({}, {word: "word", count: "count"}, function (err, docs) {
    //  todo 限制加载的链接数组必须是一开始固定好，修改为动态传入
    LimitLoad(listPageUrlArr, function (listPageUrl) {
        return reqPromise.then(function() {
            return pageLoad(host + searchPath + listPageUrl)
                .then(function (htmlStr) {
                    let $ = cheerio.load(htmlStr)
                        , listView = $("#ListView")
                    ;

                    //  内容区不为空
                    if (listView.length) {
                        //  初始化更新列表页数量
                        if (listPageUrlArr.length === 1) {
                            listPageUrlArr = $(".paginator span a");
                        }

                        //  详情选择 .a01
                        return listView.find(".a01").reduce(function (promise, item) {
                            //  读取详情页 href=/job/1842004.html
                            return pageLoad(host + $(item).atth("href"))
                                .then(function (htmlStr) {
                                    //  职位描述 .JobRequire
                                    let $ = cheerio.load(htmlStr)
                                        , jobRequireText = $(".JobRequire").text().toLowerCase()
                                    ;

                                    docs.forEach(function (item) {
                                        let has = 0
                                            , countFn = item => has = jobRequireText.indexOf(item) > -1
                                        ;

                                        item instanceof Array ? item.forEach(countFn) : countFn(item);
                                        has && item.count++;
                                    });

                                })
                                .catch(err => console.log("获取详情出错：" + err));
                        }, Promise.resolve());
                    }
                    else {
                        return Promise.resolve();
                    }
                })
                .catch(error => console.error("获取列表出错：" + error));
        });
    }, 5)
        .then(function () {
            StJobModel.create(docs, function (err, result) {
                if (err) {
                    console.log("存储数据出错：" + err);
                }
                else {
                    console.log("数据存储成功。");
                }
                db.close();
            });
        });
});
