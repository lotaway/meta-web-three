/**
 *  测试团购页面响应速度
 */
const http = require("http")
    , Promise = require("Promise")
    , requestUrl = "http://192.168.44.29:10010/mobi/cn/bulk/list/0/0/1.html"
    ;

var requestTimes = []
    , responseTime
    , count = 0
    ;

var timer = setInterval(function () {
    if (typeof count !== "number" || ++count > 2) clearInterval(timer);
    for (let i = 0; i < 10000; i++) {
        requestTimes[i] = Date.now();
        http.get(requestUrl, function (res) {

            var html = [];

            res.on('data', data => html.push(data));

            res.on('end', function () {
                console.log("第" + i.toString() + "次，响应时间：" + requestTimes[i].toString() + "， 耗时：" + (Date.now() - requestTimes[i]).toString() + "毫秒");
            });

        }).on('error', () => console.log('catch error'));
    }
}, 1000);