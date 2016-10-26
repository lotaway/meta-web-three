/**
 * Created by lotaway on 2016/5/8.
 * 发送异步请求，可跨域！
 */
var http = require('http');
var querystring = require('querystring');

//  发送的数据
var postData = querystring.stringify({
    'content': '感觉突然就与现实世界发生了奇妙的联系',
    'mid': 8837
});

//  发送的内容
var options = {
    host: 'www.imooc.com',
    port: 80,
    path: '/course/docomment',
    method: 'POST',
    headers: {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'Connection': 'keep-alive',
        'Content-Length': postData.length,
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Cookie': 'imooc_uuid=6ba12b2c-c6c1-4f74-9be4-792cb089ad88; imooc_isnew=1; imooc_isnew_ct=1462681278; IMCDNS=0; loginstate=1; apsid=FjZGZmMzMwYTNhN2QzODQ0YmVlMjBlY2Q3MDJmZDQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMjQ0MzEzMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA1NzY2OTYyOTRAcXEuY29tAAAAAAAAAAAAAAAAAAAAAGJjNTZmZGVjZDU3MjcyOGJlY2I2ZTJmODRiYTYyMGVmb78uV2%2B%2FLlc%3DYz; last_login_username=576696294%40qq.com; PHPSESSID=p1deqv8sn8jugd8cdvdvdcqb13; jwplayer.volume=47; Hm_lvt_f0cfcccd7b1393990c78efdeebff3968=1462681284,1462698306; Hm_lpvt_f0cfcccd7b1393990c78efdeebff3968=1462698332; cvde=572f014052953-10',
        'Host': 'www.imooc.com',
        'Origin': 'http://www.imooc.com',
        'Referer': 'http://www.imooc.com/video/8837',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }
};

var req = http.request(options, function (res) {
    console.log(res.statusCode);
    console.log(JSON.stringify(res.headers));

    res.on('data', function (data) {
        console.log(Buffer.isBuffer(data));
        console.log(typeof data);
    });

    res.on('end', function () {
        console.log('提交完毕');
    });

});

/*req.on('error', function () {
 console.log('错误');
 });

 req.write(postData);

 req.end();*/

req.on('error', function () {
    return console.log('错误');
});
req.write(postData);
req.end();