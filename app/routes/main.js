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
    list = require('../controllers/goods.server.controller'),
    test = require('./test'),
    handler = {
        /*
        http://www.ruanyifeng.com/blog/2014/05/restful_api.html

        GET（SELECT）：从服务器取出资源（一项或多项）。
        POST（CREATE）：在服务器新建一个资源。
        PUT（UPDATE）：在服务器更新资源（客户端提供改变后的完整资源）。
        PATCH（UPDATE）：在服务器更新资源（客户端提供改变的属性）。
        DELETE（DELETE）：从服务器删除资源。
        HEAD：获取资源的元数据。
        OPTIONS：获取信息，关于资源的哪些属性是客户端可以改变的。

        示例：
         GET /zoos：列出所有动物园
         POST /zoos：新建一个动物园
         GET /zoos/ID：获取某个指定动物园的信息
         PUT /zoos/ID：更新某个指定动物园的信息（提供该动物园的全部信息）
         PATCH /zoos/ID：更新某个指定动物园的信息（提供该动物园的部分信息）
         DELETE /zoos/ID：删除某个动物园
         GET /zoos/ID/animals：列出某个指定动物园的所有动物
         DELETE /zoos/ID/animals/ID：删除某个指定动物园的指定动物

         下面是一些常见的参数。
         ?limit=10：指定返回记录的数量
         ?offset=10：指定返回记录的开始位置。
         ?page=2&per_page=100：指定第几页，以及每页的记录数。
         ?sortby=name&order=asc：指定返回结果按照哪个属性排序，以及排序顺序。
         ?animal_type_id=1：指定筛选条件
         参数的设计允许存在冗余，即允许API路径和URL参数偶尔有重复。比如，GET /zoo/ID/animals 与 GET /animals?zoo_id=ID 的含义是相同的。
        */
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
            '/api/getui': test.api,   //  个推接口
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