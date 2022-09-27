'use strict';
/*
   从网站爬取图片
 */
const fs = require('fs'),
    path = require('path'),
    request = require('request'),
    cheerio = require('cheerio')
    , {pageLoad} = require("../nodejs/utils/loader")
;

QueryPage(
    /*设置地址*/'https://www.behance.net/search',
    /*搜索参数*/{
        ts: Math.floor(Date.now() / 1000),    //  第一页加载时的时间，固定
        ordinal: 0,     //  起始的位置
        per_page: 12,   //  每页数量
        field: 132, //  132='用户界面/用户体验‘
        content: "projects", // 筛选=['projects','users','teams']
        sort: "appreciations",  //  排序=[/*好评最多*/appreciations,/*特别推荐*/featured_date,/*查看最多*/views,/*评论最多*/comments,/*最新*/published_date]
        time: 'week',   //  时间段=[today,week,month]
        location_id: '' //  选择地区？
    }, {
        index_page: 1,
        max_page: 3,
        max_loop: 5
    });

function QueryPage(request_url, queryValue, options) {
    let query = '';

    if (options.index_page > options.max_page || options.index_page++ > options.max_loop) {
        console.log('不符合要求');
        return;
    }
    for (let key of Object.keys(queryValue)) {
        query += key + '=' + queryValue[key] + '&';
    }
    console.log(query + ' ———— 列表页请求发送');
    pageLoad(request_url + '?' + query)
        .then(body => {
            var result
                , $ = cheerio.load(body)
                , promise_array = []
            ;

            $("#content .cover-img-link").each(function () {
                result.push($(this).attr('href'));
            });

            if (!result.length)
                throw new Error("array is empty");

            for (let i = 0; i < result.length; ++i) {
                promise_array.push((function (i) {
                    return pageLoad(result[i])
                        .then(function (htmlStr) {
                            let savePath, //  要保存的路径
                                $,  //  承载解析的html内容
                                filename,
                                img_array = [];  //  图片数组

                            $ = cheerio.load(htmlStr);
                            savePath = mkdirSync(path.join('./download', i.toString(), $("#project-name").text().replace(/[/:\.\*\?"<>\|\r]+/, ' ')));
                            $("#project-modules picture img").each(function () {
                                img_array.push($(this).attr('src'));
                            });

                            return img_array.reduce(function (promise, imgUrl, i) {
                                filename = path.basename(imgUrl);

                                return promise.then(function () {
                                    return new Promise(function (resolve, reject) {
                                        request(imgUrl)
                                            .pipe(fs.createWriteStream(path.join(savePath, i.toString() + '-' + filename)))
                                            .on('close', (function (filename) {
                                                return function () {
                                                    resolve();
                                                    console.log(filename + ' DONE');
                                                }
                                            })(filename))
                                            .on('error', err => reject(err));
                                    });
                                });


                            }, Promise.resolve());
                        })
                        .catch(function (err) {
                            console.log(err ? '错误2：' + err : result[i] + '详情页请求完成');
                        });

                })(i));
            }

            Promise
                .all(promise_array)
                .then(function () {
                    console.log('正在准备新一页');
                    queryValue.ordinal += queryValue.per_page;
                    QueryPage(request_url, queryValue, options);
                });

        });
}

function mkdirSync(address) {
    var pathParts = address.split(path.sep);

    for (let i = 1; i <= pathParts.length; i++) {
        address = path.join.apply(null, pathParts.slice(0, i));
        fs.existsSync(address) || fs.mkdirSync(address);
    }
    return address;
}