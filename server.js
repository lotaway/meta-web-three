const {Nuxt, Builder} = require("nuxt")
    , app = require("express")()
    , port = process.env.PORT || 3000
;

//  传入配置初始化实例
const config = require("./nuxt.config.js")
    , nuxt = new Nuxt(config)
;

app.use(nuxt.render);
//  在开发模式下进行编译
if (config.dev) {
    process.env.DEBUG = "nuxt:*";
    new Builder(nuxt).build();
}
//  监听指定端口
app.listen(port, "0.0.0.0");
console.log("服务器运行于localhost:" + port);