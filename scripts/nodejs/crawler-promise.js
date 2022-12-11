/**
 * Promise异步 多页面 网络爬虫， 2016/5/8.
 */
const http = require('http')
    , cheerio = require('cheerio')
    , Promise = require('Promise')
    , getPageSync = require("./utils/getPage")
;

var rootUrl = 'http://www.imooc.com/u/108492/courses',
    baseUrl = 'http://www.imooc.com/learn',
    msgs = [],
    promises = [];

getPageSync(rootUrl).then(function (html) {
    var $ = cheerio.load(html);
    $('.course-one').each(function () {
        msgs.push({
            url: baseUrl + '/' + $(this).attr("data-courseid"),
            classes: []
        });
    });
    msgs.forEach(function (item) {
        promises.push(getPageSync(item.url));
    });

    Promise.all(promises).then(function (pages) {
        pages.forEach(function (html, i) {
            var $ = cheerio.load(html),
                cs = $('.chapter');
            cs.each(function () {
                var c = $(this);
                var ct = c.find('strong').text(),
                    vs = c.find('.video').children('li');
                var cd = {
                    chapterTitle: ct,
                    videos: [],
                    codes: []
                };
                vs.each(function (i) {
                    if ($(this).find('.studyvideo').length) {
                        let v = $(this).find('.studyvideo');
                        let vt = v.text();
                        let id = v.attr('href').split('video/')[1];
                        cd.videos[i] = {
                            title: vt,
                            id: id
                        };
                    }
                    else if ($(this).find(".programme").length) {
                        let c = $(this).find('.programme');
                        let ct = c.text();
                        let id = c.attr('href').split('code/')[1];
                        cd.codes[i] = {
                            title: ct,
                            id: id
                        };
                        //console.log(JSON.stringify(cd));
                    }
                });
                msgs[i].classes.push(cd);
            });
        });
        console.log(msgs);
    }, e => console.log(e));
}, e => console.log(e));