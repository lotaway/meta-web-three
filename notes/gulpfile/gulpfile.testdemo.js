/**
 * http://web.jobbole.com/83340/   前端工程简介
 * http://www.gulpjs.com.cn/docs/api/ gulp中文网
 */
var gulp = require('gulp');
var sass = require('gulp-sass');    //  编译sass
var cache = require('gulp-cache');  //  只对有修改的文件执行任务，其他读取缓存
//gulpBabel = require('gulp-babel'),  //  es6转es5
var webpack = require('gulp-webpack');  //  构建工具，用于合并js文件，解决6转5后的依赖问题
var rev = require('gulp-rev');          //  对静态文件名称附加时间戳并生成对应关系表
var revCollector = require('gulp-rev-collector');   //  根据关系表替换html文件里的引用路径
var spriter = require('gulp-css-spriter');   //  css背景图合成
var imagemin = require('gulp-imagemin');   //  图片压缩，还有子模块用于深度压缩
//var minifyHTML = require('gulp-minify-html');
//var plugins=require('gulp-load-plugins')(); //  可以通过它调用gulp其他插件，但先要引入其他插件？
//var jshint = require("gulp-jshint");    //  js代码检查
//var del = require('del');                    // 文件删除
//var concat = require("gulp-concat");    //  合并js/css文件
//var uglify = require('gulp-uglify');    //  js压缩
//var minifyCss = require("gulp-minify-css"); //  css压缩
//var minifyHtml = require('gulp-minify-html');   //  html压缩
//var rename = require('gulp-rename');    //  文件可重命名
//var less = require('gulp-less');    //  less文件编译
//var imagemin = require('gulp-imagemin');    //  压缩图片
//var pngquant = require('imagemin-pngquant');    //  压缩png
//var livereload = require('gulp-livereload');    //  用于（文件修改时）刷新页面

//  测试
/*gulp.task('default', function () {
    gulp
        .src('app/routes/!*.js', {
            base: 'app'
        })
        //.pipe(rename('newName.js'))
        //.pipe(uglify())
        //.pipe(concat('allToOne.js'))
        .pipe(gulp.dest('gulpTest'));

    //gulp.run()

    gulp.start('watch_reload', 'watch_jade');

});*/

//  jade预编译
gulp.task('watch_jade', function () {
    gulp.watch('views/*.jade', function (event) {
        console.log(event.path + " is " + event.type);
    });
});
//  less预编译
gulp.task('less', function () {
    gulp.src('less/*.less')
        .pipe(less())
        .pipe(gulp.dest('css'))
        .pipe(livereload());
});
//  重载浏览器
gulp.task('watch_reload', function () {
    livereload.listen(); //要在这里调用listen()方法
    gulp.watch('less/*.less', ['less']);
});

//  监视：es6.js文件
gulp.task("watch-es6", function () {
    gulp.watch("unit_test/es6/*.js", ['es6-to-5'])
});
//  jsx/es6转es5
gulp.task("babel-transform", function () {
    browserify('unit_test/es6/**/*.js')
        .transform(babelify, {
            presets: ['es2015', 'react']
        })
        .bundle()
        .pipe(gulp.dest("unit_test/react/transitions"))
        .pipe(webpack({
                //babel编译import会转成require，webpack再包装以下代码让代码里支持require
                output: {
                    filename: "bundle.js"
                },
                stats: {
                    colors: true
                }
            }
        ))
        .pipe(gulp.dest("unit_test"));
});

//  对css文件内的图标资源合并，并压缩
gulp.task('css-images-sprite', function () {
    //   合并图片
    gulp.src("./template/css/*.css")
        .pipe(spriter({
            spriteSheet: './dist/images/icon.png', //  合成后生成的图片的存放路径
            pathToSpriteSheetFromCSS: '../images/icon.png' //  合成后图片在css中的引用路径
        }))
        .pipe(gulp.dest('./dist')); //  合成后css文件的路径

//  图片压缩
    gulp.src('./dist/images/*')
        .pipe(cache(imagemin(
            {
                optimizationLevel: 5, //类型：Number  默认：3  取值范围：0-7（优化等级）
                progressive: true, //类型：Boolean 默认：false 无损压缩jpg图片
                interlaced: true, //类型：Boolean 默认：false 隔行扫描gif进行渲染
                multipass: true //类型：Boolean 默认：false 多次优化svg直到完全优化
            }
        )))
        .pipe(gulp.dest('./dist'));
});

//  文件名附加时间戳
gulp.task('name-timestamp', function () {
    //  处理css文件
    gulp.src(['./template/**/**/*.css'], {base: './template'})
        .pipe(rev())
        .pipe(gulp.dest('./dist'))
        .pipe(rev.manifest())
        .pipe(gulp.dest('./dist/rev/css'));
    //  处理js文件
    gulp.src(['./template/**/**/*.js'], {base: './template'})
        .pipe(rev())
        .pipe(uglify())
        .pipe(gulp.dest('./dist'))
        .pipe(rev.manifest())
        .pipe(gulp.dest('./dist/rev/js'));

//  html内引用文件替换
    return gulp.src(['./dist/rev/**/*.json', './template/**/**/*.html'])
        .pipe(revCollector({
            replaceReved: true,
            /*dirReplacements: {
             'css': 'dist\/css',
             'js': 'dist\/js',
             'cdn/': function (manifest_value) {
             return '//cdn' + (Math.floor(Math.random() * 9) + 1) + '.' + 'exsample.dot' + '/img/' + manifest_value;
             }
             }*/
        }))
        /*.pipe(minifyHTML({
         empty: true,
         spare: true
         }))*/
        .pipe(gulp.dest('./dist'));
});