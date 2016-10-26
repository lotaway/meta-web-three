/*
 工具来源：

 Nodejs需要手动加载路由文件，如果一个个添加，项目逐渐扩大，比较麻烦。
 尤其在项目route目录下，增加模块文件夹的时候，引入路由更是麻烦。
 因此写了一个Nodejs动态加载路由，Nodejs遍历目录，Nodejs路由工具，取名为route.js。

 支持无限级别目录结构，自动递归引用。有任何更好的建议，欢迎随时留意交流。

 使用方法：

 1、文件：app.js同级目录增加route.js文件，复制下面贴出源代码；
 2、引入：app.js中引入：var route = require('./route');
 3、调用：在app变量初始化之后，在app.js中使用route.init(app,[可选参数，路由目录，默认为./routes/])，即可动态加载路由文件了;
 */

/**
 * 动态遍历目录加载路由工具
 * author: bling兴哥
 */
var fs = require("fs");
// 动态路由
var loadRoute = {
    path: './routes/',
    app: null,
    // 遍历目录
    listDir: function (dir) {
        var fileList = fs.readdirSync(dir, 'utf-8');
        for (var i = 0; i < fileList.length; i++) {
            // 是目录则继续，否则进行路由加载
            if (fs.lstatSync(dir + fileList[i]).isDirectory()) {
                this.listDir(dir + fileList[i] + '/');
            }
            else {
                this.loadRoute(dir + fileList[i]);
            }
        }
    },
    // 加载路由
    loadRoute: function (routeFile) {
        console.log(routeFile);
        var route = require(routeFile.substring(0, routeFile.lastIndexOf('.')));
        // 在路由文件中定义了一个basePath变量，设置路由路径前缀
        if (route.basePath) {
            this.app.use(route.basePath, route);
        }
        else {
            this.app.use(route);
        }
    },
    // 初始化入口
    init: function (app, path) {
        if (!app) {
            console.error("系统主参数App未设置");
            return false;
        }
        this.app = app;
        this.path = path ? path : this.path;
        this.listDir(this.path);
    }
};

module.exports = loadRoute;