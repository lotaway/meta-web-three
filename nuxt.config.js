// 参考 https://zh.nuxtjs.org/api/configuration-build/#babel
var webpack = require("webpack");

module.exports = {
    //  模块开发扩展核心功能，参考 https://zh.nuxtjs.org/guide/modules
    modules: [
        // '@nuxtjs/style-resources',
    ]
    , build: {
        //  使用extract-css-chunks-webpack-plugin启用CSS提取
        extractCSS: true
        // 调用webpack-bundle-analyzer分析打包：https://github.com/webpack-contrib/webpack-bundle-analyzer#as-plugin
        /*analyze: {
            analyzerMode: 'static'
        },*/
        , babel: {
            presets: [
                "es2015"
                , "stage-0"
            ]
        }
        , extend(config, {isDev, isClient, isServer, loaders: {vue}}) {
            //  为客户端打包进行扩展配置
            if (isClient) {
                config.devtool = "eval-source-map";
                vue.transformAssetUrls.video = ['src', 'poster'];
            }
        }
        //  缓存
        /*,cache: {
            max: 1000,
            maxAge: 900000
        },*/
        //  在生成的HTML中的<link rel ="stylesheet">和<script>标签上配置crossorigin属性。
        // , crossorigin: String
        // , cssSourceMap: true
        //  自定义打包文件名
        /*, filenames: {
            app: ({isDev}) => isDev ? '[name].js' : '[chunkhash].js',
            chunk: ({isDev}) => isDev ? '[name].js' : '[chunkhash].js',
            css: ({isDev}) => isDev ? '[name].css' : '[contenthash].css',
            img: ({isDev}) => isDev ? '[path][name].[ext]' : 'img/[hash:7].[ext]',
            font: ({isDev}) => isDev ? '[path][name].[ext]' : 'fonts/[hash:7].[ext]',
            video: ({isDev}) => isDev ? '[path][name].[ext]' : 'videos/[hash:7].[ext]'
        }*/
        // , hotMiddleware: {}
        //  用于压缩在构建打包过程中创建的HTML文件配置html-minifier的插件
        /*, htmlMinify: {
            collapseBooleanAttributes: true,
            collapseWhitespace: false,
            decodeEntities: true,
            minifyCSS: true,
            minifyJS: true,
            processConditionalComments: true,
            removeAttributeQuotes: false,
            removeComments: false,
            removeEmptyAttributes: true,
            removeOptionalTags: false,
            removeRedundantAttributes: true,
            removeScriptTypeAttributes: false,
            removeStyleLinkTypeAttributes: false,
            removeTagWhitespace: false,
            sortClassName: false,
            trimCustomFragments: true,
            useShortDoctype: true
        }*/
        //  自定义 webpack 加载器
        /*, loaders: {
            file: {},
            fontUrl: { limit: 1000 },
            imgUrl: { limit: 1000 },
            pugPlain: {},
            vue: {
                transformAssetUrls: {
                    video: 'src',
                    source: 'src',
                    object: 'src',
                    embed: 'src'
                }
            },
            css: {},
            cssModules: {
                localIdentName: '[local]_[hash:base64:5]'
            },
            less: {},
            sass: {
                indentedSyntax: true
            },
            scss: {},
            stylus: {},
            vueStyle: {}
        }*/
        /*//  css资源选项插件，参考 https://github.com/NMFR/optimize-css-assets-webpack-plugin
        , optimizeCSS: false*/
        /*//  webpack构建打包中开启 thread-loader，参考 https://github.com/webpack-contrib/thread-loader#thread-loader
        , parallel: false*/
        //  配置webpack插件
        , plugins: [
            new webpack.DefinePlugin({
                "process.VERSION": "1.1.20190201"
            })
        ]
        /*//  样式后置处理，Nuxt.js已应用PostCSS Preset Env。默认情况下，它将启用Stage 2功能和Autoprefixer,你可以使用build.postcss.preset来配置它。
        , postcss: {
            plugins: {
                'postcss-import': {},
                'postcss-url': {},
                'postcss-preset-env': {},
                'cssnano': {preset: 'default'} // disabled in dev mode
            }
        }*/
        /*//  可以通过在命令行中加入 --profile 来开启webpackBar，参考 https://github.com/nuxt/webpackbar#profile
        , profile: false*/
        /*//  公共路径，如/.nuxt/，一般用于设置CDN
        , publicPath: "https://cdn.nuxtjs.org"*/
        /*//  控制部分构建信息输出日志，默认:检测到CI或test环境时启用 std-env
        , quiet: false*/
        /*//  代码分模块，常用: vue|vue-loader|vue-router|vuex...
        , splitChunks: {
            layouts: false,
            pages: true,
            commons: true
        }*/
        /*//  默认: true 为通用模式，false 为spa模式
        , ssr: true*/
        /*//  预置全局样式，当您需要在页面中注入一些变量和mixin而不必每次都导入它们时。参考 https://github.com/nuxt-community/style-resources-module
        , styleResources: {
            sass: "./assets/variables.sass"
        }*/
        /*// 自定义自己的模板，这些模板将基于Nuxt配置进行渲染。
        , templates: [
            {
                src: '~/modules/support/plugin.js', // `src` 可以是绝对的或相对的路径
                dst: 'support.js', // `dst` 是相对于项目`.nuxt`目录
                options: { // 选项`options`选项可以将参数提供给模板
                    live_chat: false
                }
            }
        ]*/
        /*//  使用Babel与特定的依赖关系进行转换，选项可以是字符串或正则表达式对象，用于匹配依赖项文件名
        , transpile: []*/
        /*//  监听文件更改。
        , watch: [
            '~/.nuxt/support.js'
        ]*/
    }
    /*//  构建输出目录，默认为.nuxt
    , buildDir: "dist"*/
    /*//  后置全局样式
    , css: [
        // 直接加载一个 Node.js 模块。（在这里它是一个 Sass 文件）
        'bulma',
        // 项目里要用的 SASS 文件
        '@/assets/css/main.sass'
    ]*/
    /*//  开启开发模式，nuxt命令会覆盖此选项
    , dev: true*/
};