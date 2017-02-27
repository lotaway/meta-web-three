/**
 * Created by lotaway on 2016/5/11.
 */
var express = require('express'),
    router = express.Router(),
    url = require('url'),
    index = require('./index'),
    user = require('../controllers/user.server.controller'),
    download = require('./download'),
    search = require('./search'),
    out = require('./out'),
    list = require('../controllers/list.server.controller'),
    test = require('./test'),
    handler = {
        GET: {
            '/': index.start,           //  首页
            '/test/index': test.start,   //  任意测试
            '/test/getui': test.getui,   //  个推
            '/list': list.start,        //  商品列表
            //'/list/:id': list.start,  //  商品详情
            '/reg': user.signIn,     //  用户注册
            '/login': user.loginIn,      //  用户登录
            //'/user/login': login.form,//  用户登录表单提交
            '/user': user.start,        //  用户个人中心
            '/search': search.start,    //  搜索页
            '/download': download.start,    //  下载
            '/out': out.start           //  登出
        },
        POST: {
            '/user/signUp': user.signUp,
            '/post/search': search.post
        }
    };

//  使用中间件
router.use(function (req, res, next) {
    var pathname = url.parse(req.url).pathname;
    console.log(pathname + ' 请求接收于：' + Date.now());

//  把response对象传给handlers模块，直接将结果返回到页面或进行处理。
    req.setTimeout(1000, function (req, res) {
        console.log('响应超时');
        res.redirect('/404');
    });
    //  判断当前映射的action是否为一个函数
    if (typeof handler[req.method][pathname] === 'function') {
        //  直接执行handle
        var content = handler[req.method][pathname](req, res, next);
        console.log("handle content: " + content);  //  处理返回的内容，可以用于？
    }
    else {
        /*console.log("No request handler found for " + pathname);
         res.write(404, {"Content-Type": "text/plain"});
         res.write("404 not found");
         res.end();*/
    }

    //next();
});

module.exports = router;