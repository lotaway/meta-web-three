let webpack = require('webpack');
//var commonsPlugin = new webpack.optimize.CommonsChunkPlugin('common.js');
/*var htmlWebpackPlugin = require('html-webpack-plugin'); //使用自动生成html文件的一个插件
var path = require('path');*/

module.exports = {
    entry: {
        index: './unit_test/es6/react/transitions/',
        //entry2: './entry/entry2.js'
    },
    output: {
        path: "./unit_test/es6/react/transitions/",
        filename: '[name].entry.js'
    },
    resolve: {
        extensions: ['', '.js', '.jsx']
    },
    module: {
        loaders: [{
            test: /\.js$/,
            loader: 'babel-loader'
        }, {
            test: /\.jsx$/,
            loader: 'babel',
            exclude:"/node_modules/",
            //include:path.resolve(__dirname,"example")
        }]
    },
    /*devServer: {
        hot:true,
        inline:true
    },*/
    /*plugins: [commonsPlugin],
    new htmlWebpackPlugin({
        title:"First react app"
    })*/
};
/*
这里对Webpack的打包行为做了配置，主要分为几个部分：

entry：指定打包的入口文件，每有一个键值对，就是一个入口文件
output：配置打包结果，path定义了输出的文件夹，filename则定义了打包结果文件的名称，filename里面的[name]会由entry中的键（这里是entry1和entry2）替换
resolve：定义了解析模块路径时的配置，常用的就是extensions，可以用来指定模块的后缀，这样在引入模块时就不需要写后缀了，会自动补全
module：定义了对模块的处理逻辑，这里可以用loaders定义了一系列的加载器，以及一些正则。当需要加载的文件匹配test的正则时，就会调用后面的loader对文件进行处理，这正是webpack强大的原因。比如这里定义了凡是.js结尾的文件都是用babel-loader做处理，而.jsx结尾的文件会先经过jsx-loader处理，然后经过babel-loader处理。当然这些loader也需要通过npm install安装
plugins: 这里定义了需要使用的插件，比如commonsPlugin在打包多个入口文件时会提取出公用的部分，生成common.js
当然Webpack还有很多其他的配置，
具体可以参照它的配置文档http://webpack.github.io/docs/configuration.html#entry
 https://segmentfault.com/a/1190000002767365
*/
