'use strict';

const fs = require('fs'),
    path = require('path'),
    request = require('request'),
    cheerio = require('cheerio');

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

    var query = '';

    if (options.index_page > options.max_page || options.index_page++ > options.max_loop) {
        console.log('不符合要求');
        return;
    }
    for (let key of Object.keys(queryValue)) {
        query += key + '=' + queryValue[key] + '&';
    }

    console.log(query + ' ———— 列表页请求发送');

    new Promise((resolve, rejected) => {

        request(request_url + '?' + query, function (err, res, body) {

            console.log(err ? '错误1：' + err : query + '列表页面请求完成');

            var $,
                link_array = [];

            if (!err && res.statusCode == 200) {
                $ = cheerio.load(body);
                $("#content .cover-img-link").each(function () {
                    link_array.push($(this).attr('href'));
                });
                link_array.length ? resolve(link_array) : rejected('array is empty');
            }
        });
    })
        .then(result => {

            if (!result.length) return null;

            var promise_array = [];

            for (let i = 0; i < result.length; ++i) {
                promise_array.push((function (i) {
                    return new Promise(function (resolve, rejected) {
                        request(result[i], function (err, res, body) {
                            var savePath, //  要保存的路径
                                $,  //  承载解析的html内容
                                filename,
                                img_array = [];  //  图片数组

                            if (err) return rejected(err);
                            console.log(err ? '错误2：' + err : result[i] + '详情页请求完成');
                            if (res.statusCode == 200) {
                                $ = cheerio.load(body);
                                savePath = mkdirSync(path.join('./download', i.toString(), $("#project-name").text().replace(/[/:\.\*\?"<>\|\r]+/,' ')));
                                $("#project-modules picture img").each(function () {
                                    img_array.push($(this).attr('src'));
                                });
                                for (let i = 0; i < img_array.length; i++) {
                                    filename = path.basename(img_array[i]);
                                    request(img_array[i])
                                        .pipe(fs.createWriteStream(path.join(savePath, i.toString() + '-' + filename)))
                                        .on('close', (function (filename) {
                                            return function () {
                                                resolve();
                                                console.log(filename + ' DONE');
                                            }
                                        })(filename))
                                        .on('error', err => rejected(err));
                                }

                            }
                        });
                    })
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