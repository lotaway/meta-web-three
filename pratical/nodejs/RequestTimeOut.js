/**
 * Created by lotaway on 2016/6/24.
 */

/*下表中响应对象（res）的方法向客户端返回响应，终结请求响应的循环。如果在路由句柄中一个方法也不调用，来自客户端的请求会一直挂起。
 方法	             描述
 res.download()	    提示下载文件。
 res.end()	        终结响应处理流程。
 res.json()	        发送一个 JSON 格式的响应。
 res.jsonp()	    发送一个支持 JSONP 的 JSON 格式的响应。
 res.redirect()	    重定向请求。
 res.render()	    渲染视图模板。
 res.send()	        发送各种类型的响应。
 res.sendFile	    以八位字节流的形式发送文件。
 res.sendStatus()	设置响应状态代码，并将其以字符串形式作为响应体的一部分发送。*/

/*如果在指定的时间内服务器没有做出响应(可能是网络间连接出现问题,也可能是因为服务器故障或网络防火墙阻止了客户端与服务器的连接),则响应超时,同时触发http.ServerResponse对象的timeout事件.
response.setTimeout(time,[callback]);
也可以不在setTimeout中指定回调函数,可以使用时间的监听的方式来指定回调函数.
    如果没有指定超时的回调函数,那么出现超时了,将会自动关闭与http客户端连接的socket端口.如果指定了超时的回调函数,那么超时了,将会出现调用回调函数,而不会自动关闭与http客户端连接的socket端口.
    复制代码 代码如下:
 */
var http = require("http");
var server = http.createServer(function (req, res) {
    if (req.url !== "/favicon.ico") {
        res.setTimeout(1000);
        //1、监听方式完成超时响应
        res.on("timeout", function () {
            console.log("响应超时.");
        });
        //2、回调方式完成超时响应
        res.setTimeout(1000, function () {
            console.log("响应超时.");
        });
        //  3、使用延时函数模仿正常情况下访问数据库再返回数据
        setTimeout(function () {
            res.setHeader("Content-Type", "text/html");
            res.write("<html><head><meta charset='utf-8' /></head>");
            res.write("你好");
            res.end();
        }, 2000);
    }
});
server.listen(1337, "localhost", function () {
    console.log("开始监听" + server.address().port + "......");
});