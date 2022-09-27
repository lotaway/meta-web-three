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
        'Content-Length': postData.length,
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Cookie': '',
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

req.on('error', function () {
    return console.log('错误');
});
req.write(postData);
req.end();