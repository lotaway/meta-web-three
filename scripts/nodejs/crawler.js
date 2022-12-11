/**
 * 网络爬虫，简单获取页面内容 2016/5/8.
 */
var http = require('http');
var cheerio = require('cheerio');

var url = 'http://www.imooc.com/learn/348';

function filterHtml(html) {
    var $ = cheerio.load(html),
        cds = [];

    var cs = $('.chapter');
    cs.each(function (item) {
        var c = $(this);
        var ct = c.find('strong').text(),
            vs = c.find('.video').children('li');
        var cd = {
            chapterTitle: ct,
            videos: []
        };
        vs.each(function (item) {
            var v = $(this).find('.studyvideo');
            var vt = v.text();
            var id = v.attr('href').split('video/')[1];
            cd.videos.push({
                title: vt,
                id: id
            });
        });
        cds.push(cd);
    });

    return cds;
}

http.get(url, function (res) {

    var html = [];

    res.on('data', data => html.push(data));

    res.on('end', function () {
        html = html.join('');
        var h = filterHtml(html);
        console.log(h);
    });

}).on('error', () => console.log('catch error'));