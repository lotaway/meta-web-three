const http = require("http"),
    https = require("https"),
    domain = require("domain")
path = require("path"),
    querystring = require("querystring"),
    fs = require("fs"),
    open = require("open"),
    cheerio = require("cheerio"),
    iconv = require("iconv-lite"),
    targetChapterLimit = 50 //  读取的最新章节数量
    ,
    port = 3000 //	visit url port
    , domainGetNovel = domain.create();

//  todo 对于站点会自行判断userAgent引发的问题，需要设置userAgent？
domainGetNovel.run(() => {
    let server = http.createServer((req, res) => {
        let template = data => data.list.reduce((prev, item) => {
                let linkStr = (item.link && item.link.length ? item.link : []).reduce((prev, cur, index) => prev + ' <a href="' + cur + '">链接' + (1 + index) + '</a>', '');

                return prev + `<li><label>${item.title || "未知"}</label>${linkStr}</li>`
            }, '<ul>') + '</ul>',
            reqData = "";

        req.on("data", chunk => {
            reqData += chunk;
        });
        req.on("end", () => {
            const router = {
                    "/getNovel": (req, res, body) => {
                        let targetWebsite = [{
                            host: "https://c.biduo.cc",
                            pathname: "/biquge/59_59347/",
                            encoding: "binary",
                            complier: (data, {host}) => {
                                const $ = cheerio.load(iconv.decode(data, "GBK"));
                                let links = {list: []};

                                $(".chapter a").each((index, aNode) => {
                                    if (index < targetChapterLimit) {
                                        links.list.push({
                                            title: $(aNode).text(),
                                            link: host + $(aNode).attr("href")
                                        });
                                    }
                                });

                                return links;
                            }
                        },
                            {
                                host: "https://m.jx.la",
                                pathname: '/book/254285/',
                                encoding: "utf8",
                                complier: (data, {host}) => {
                                    const $ = cheerio.load(data);
                                    let links = {list: []};

                                    console.log($("#chapterlist").length);
                                    $("#chapterlist a").each((index, aNode) => {
                                        if (index < targetChapterLimit) {
                                            links.list.push({
                                                title: $(aNode).text(),
                                                link: host + $(aNode).attr("href")
                                            });
                                        }
                                    });

                                    return links;
                                }
                            }
                        ];

                        res.writeHeader(200, {"Content-Type": "text/html; charset=utf-8"});
                        Promise.all(targetWebsite.map(item => {
                            return new Promise((resolve, reject) => {
                                https.get(item.host + item.pathname, res => {
                                    let htmlStr = '';

                                    res.setEncoding(item.encoding);
                                    res.on("data", chunk => {
                                        htmlStr += chunk;
                                    });
                                    res.on("end", () => {
                                        resolve(item.complier(htmlStr, {
                                            host: item.host
                                        }));
                                    });
                                }).on("error", err => {
                                    reject(err + "，with：" + item.host + item.pathname);
                                }).on("finish", () => {
                                    console.log("进入结束事件");
                                });
                            });
                        })).then(data => {
                            console.info(data);
                            const filePath = path.join(__dirname, "./temp/getNovelLog.json");
                            let finalData = {list: []},
                                ungroup = [],
                                matcher = {};

                            data.forEach((one, oneIndex) => {
                                if (ungroup.length) {
                                    const lastIndex = Object.keys(matcher).length;

                                    ungroup.forEach((item, index) => addMatcher[item.title, item.link, lastIndex + index]);
                                    ungroup = [];
                                }
                                one.list.forEach((item, itemIndex) => {
                                    const addMatcher = (title, link, index) => {
                                        matcher[title] = {
                                            index,
                                            title,
                                            link: [link]
                                        };
                                    }

                                    if (oneIndex === 0) {
                                        addMatcher(item.title, item.link, itemIndex);
                                    } else {
                                        if (matcher[item.title]) {
                                            matcher[item.title].link.push(item.link);
                                            if (ungroup.length) {
                                                matcher = Object.values(matcher).map(item => {
                                                    item.index += ungroup.length;

                                                    return item;
                                                });
                                                ungroup.forEach((uItem, uIndex) => addMatcher(uItem.title, uItem.link, uIndex));
                                                ungroup = [];
                                            }
                                        } else {
                                            ungroup.push(item);
                                        }
                                    }
                                });
                            });
                            finalData.list = new Array(Object.keys(matcher).length);
                            Object.values(matcher).forEach(item => {
                                finalData.list[item.index] = {
                                    title: item.title,
                                    link: item.link
                                };
                            });
                            fs.mkdir(path.dirname(filePath), {recursive: true}, err => {
                                if (err) {
                                    console.error(err);
                                } else {
                                    fs.writeFile(filePath, JSON.stringify(finalData), {flag: "a"}, err => {
                                        if (!err) {
                                            console.log("写入完成：" + filePath);
                                        } else {
                                            console.error("写入出错：" + err);
                                        }
                                    });
                                }
                            });
                            res.write(template(finalData));
                            res.end();
                        }).catch(err => {
                            res.end("获取书籍数据Promise错误：" + err);
                        });
                    },
                    "/error": (req, res, body) => {
                        res.end("error, cannot found page, 404.");
                    }
                },
                body = querystring.parse(reqData);

            (req.url === "/" || !req.url) && (req.url = "/getNovel");
            console.log(req.url);
            router[router[req.url] ? req.url : "/error"](req, res, body);
        });
        req.on("error", err => {
            res.end("request err:" + err);
        });
        req.on("finish", () => {
            res.end("finish");
        });
    });

    server.listen(port);
    open(`http://localhost:${port}/getNovel`);

    console.log("excuter end");
});
