const Koa = require("koa");

const app = new Koa();

//  koa是一个中间件处理的框架，通过app.use(function)定义中间件并传入上下午和下一个中间件方法，如此在中间件A只中，可以等待后续的BC执行回调后返回执行后续处理，达成`>`的级联方式

//  日志
app.use(async (ctx, next) => {
    console.log(`开始日志处理，访问路径为：${ctx.request.method + ctx.request.url}`);
    await next();   //  等待下一个中间件执行，没有await将不会等待，没有调用next则不再执行后续中间件。
    console.log("结束日志处理");
});

//  记录响应时间
app.use(async (ctx, next) => {
    console.log("开始记录时间");
    const start = new Date().getTime();
    await next();
    const ms = new Date().getTime() - start;
    console.log(`结束记录时间：${ms}ms`);
});

//  设置cookie
app.use(async (ctx, next) => {
    ctx.cookies.set("name", "value", {
        // expires: 
    });
});

//  处理响应内容
app.use(async (ctx, next) => {
    console.log('开始处理响应');
    await next();
    //  检查请求头Content-Type类型是否符合要求
    if (ctx.is("text/html")) {
        //  最后响应的内容
        ctx.response.type = "text/html";
        ctx.response.body = "<h1>欢迎使用Koa</h1>";
    }
    else {
        //  重定向
        // ctx.redirect("back","/index.html");
        //  抛出错误
        ctx.throw(415, "错误请求类型", { msg: "更多信息" });
    }
    /*
    绕过 Koa 的 response 处理是 不被支持的. 应避免使用以下 node 属性：
    res.statusCode
    res.writeHead()
    res.write()
    res.end()
     */
    console.log("结束处理响应");
});

//  在出现错误时会调用`error`事件，主要用于集中日志记录，原本的响应流程不受此干扰
app.on("error", err => {
    console.log("服务错误：" + err);
});

/**
 * 监听相当于
 * const http = require("http")
 * http.createServer(app.callback()).listen("3002");
 * 因此可以创建多个实例或者一个实例监听http/https两个端口
 */
app.listen("3003");