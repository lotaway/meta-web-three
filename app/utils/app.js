import {initConfig} from '../config/init_config';
import {insetConfig} from '../config/base_config';
// import EventEmitter from './EventEmitter';

/**
 * 检查元素是否拥有样式名
 * @param className {String} 样式名
 * @returns {boolean} 是否拥有该样式名
 */
Element.prototype.hasClass = function (className) {
    return new RegExp("(^|\\s)" + className + "(\\s|$)").test(this.className);
};

/**
 * 把样式名赋给元素
 * @param className {String} 样式名
 * @return {Object} 元素
 */
Element.prototype.addClass = function (className) {
    !this.hasClass(className) && (this.className += " " + className);
    return this;
};

/**
 * 从元素中移除样式名
 * @param className {String} 样式名
 * @return {Object} 元素
 */
Element.prototype.removeClass = function (className) {
    this.hasClass(className) && (this.className = this.className.replace(new RegExp("((^|\\s)" + className + ")|(" + className + "(\\s|$))"), ""));
    return this;
};

/**
 * 对元素切换样式名
 * @param className {String} 样式名
 * @return {Object} 元素
 */
Element.prototype.toggleClass = function (className) {
    return this.hasClass(className) ? this.removeClass(className) : this.addClass(className);
};

/**
 * app里无归类方法
 */
class App {

    constructor() {
    }

    /**
     * 等待html5+增强接口准备好
     * @param {Function} callback 接口准备好时回调
     */
    static plusReady(callback) {
        if (window.plus) callback();
        else document.addEventListener('plusready', callback);
    }

    /**
     * 获取配置的内容
     * @param {String} name 配置名
     * @returns {*} value 配置值
     */
    static getConfig(name) {
        let value = null;
        /*if (App.storage(name)) value = App.storage(name);
         else */
        if (initConfig[name]) value = initConfig[name];
        else if (insetConfig[name]) value = insetConfig[name];
        else App.errorHandler("找不到配置项");
        return value;
    }

    /**
     * 输出信息
     * @param info {string} 信息
     * @param isMustLog {boolean} 必须输出到控制台
     */
    static logInfo(info, isMustLog = false) {
        //  只在开发环境中输出 todo 非调试环境下输出到文件里
        if (App.getConfig("environment") === "dev" || isMustLog === true) {
            // alert(info);
            console.info(info + "\n————时间：" + (new Date()).getTime());
        }
        else {

        }
    }

    /**
     * 异常处理
     * @param {Object|String} error 异常内容
     * @param {Function} next 后置处理方法
     */
    static errorHandler(error, next = null) {
        var strError;
        try {
            strError = JSON.stringify(error);
        }
        catch (e) {
            strError = error;
        }
        App.logInfo("异常：" + strError);
        if (typeof next === "function") {
            next(error);
        }
        else {
            plus.nativeUI.toast(strError);
        }
    }

//  分区加载动画
    static SectionLoader(target, after) {
        //  target为样式名：animation，包含加载前的样式，animationAfter为加载后的样式
        let self = this,
            array;

        self.array = [];

        if (!target) return;
        if (!after) after = 'animationAfter';

        array = document.querySelectorAll('.' + target);
        for (let i = 0; i < array.length; i++) {
            self.array.push(array[i]);
        }

        self.func = {
            init: function () {
                /*t.func.check();
                 $(window).scroll(function(){
                 t.func.check();
                 })*/
                for (let i = 0; i < self.array.length; i++) {
                    document.querySelectorAll(self.array[i]).removeClass(target);
                }
            },
            check: function () {
                if (!self.array.length) return;
                if (self.timer) clearTimeout(self.timer);
                self.timer = setTimeout(function () {
                    if (self.array.length) {
                        while (self.array.length) {
                            if (self.array[0].offsetTop + 100 < document.body.scrollTop + document.body.clientHeight) {
                                document.querySelectorAll(self.array.shift()).addClass(after);
                            }
                            else break;
                        }
                    }
                }, 16);
            }
        };
        self.func.init();
    }

    /**
     * 获取url参数值
     * @param {String} name 参数名
     * @returns {String|null} 参数值
     */
    static getQueryString(name) {
        if (!name) return null;
        //构造一个含有目标参数的正则表达式对象匹配目标参数
        const r = window.location.search.substr(1).match(new RegExp("(^|&)" + name + "=([^&]*)(&|$)"));
        return r === null ? r : decodeURI(r[2]);
    }

    /**
     * 读取文件
     * @param {String} filePath 文件完整路径
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 失败回调
     * @returns {Object} 文件内容、文件系统操作对象
     */
    static readFile(filePath, successCB, errorCB) {
        let paths = filePath.split('/'),    //  路径分段数组
            topDir = paths.shift(); //  路径前缀

        switch (true) {
            case /^_www/.test(topDir):
                topDir = plus.io.PRIVATE_WWW;
                break;
            case /^_doc/.test(topDir):
                topDir = plus.io.PRIVATE_DOC;
                break;
            case /^_documents/.test(topDir):
                paths[0] = plus.io.PUBLIC_DOCUMENTS;
                break;
            case /^_download/.test(topDir):
                topDir = plus.io.PUBLIC_DOWNLOADS;
                break;
            default:
                return errorCB('没有指定顶级目录');
        }
        plus.io.requestFileSystem(topDir, fs => {
            getDirectory(paths, fs.root, (file, fileEntry) => {
                let fileReader = new plus.io.FileReader();
                fileReader.onloadend = function (e) {
                    successCB(e, fileEntry);
                };
                fileReader.onerror = errorCB;
                fileReader.readAsText(file);
            });

        });

        /**
         * 创建并进入目录
         * @param paths 单层路径
         * @param entry 文件系统操作对象
         * @param callback 回调
         */
        function getDirectory(paths, entry, callback) {
            let path = paths.shift(), //  路径前缀
                nextCall;   //  下一个处理方法

            nextCall = paths.length > 1 ? getDirectory : getFile;
            entry.getDirectory(path, {
                create: true
            }, nextCall(paths, entry, callback));
        }

        /**
         * 创建并打开文件（单级）
         * @param fileChunk 文件全名
         * @param entry 文件系统操作对象
         * @param callback 回调
         */
        function getFile(fileChunk, entry, callback) {
            entry.getFile(fileChunk, {create: true}, function (fileEntry) {
                fileEntry.file(fs => {
                    callback(fs, fileEntry);
                });
            });
        }
    }

    /**
     * 将预定的超链接转为带域名的链接
     * @param tag {String|array} 匹配标签|匹配标签数组
     * @param attrName {String} 属性名称
     */
    static transformUrl(tag = "a", attrName = "href") {
        var targetArray = typeof tag === "string" ? document.querySelectorAll(tag) : tag;
        Array.prototype.forEach.call(targetArray, function (item) {
            if (item.getAttribute(attrName)) {
                item.setAttribute(attrName, item.getAttribute(attrName).replace(new RegExp("^[\\w\\W]*_http/"), App.getConfig('host') + "/"));
            }
        });
    }

    /**
     * 判断页面是否需要登录
     * @param url 访问的页面地址
     * @returns {Boolean} 是否需要登录
     */
    static needLogin(url) {
        return url.match("(?:(?:^|/)user_\\w+.html$)|(?:/mobi/cn/member/(\\d+)?)|(?:/mobi/cn/shopping/cart.html)|/mobi/cn/(?:\\w+/)?order/submit/\\w*\.html") !== null && !App.hasLogin();
    }

    /**
     * 判断是否已经登录
     * @returns {Boolean} 是否登录
     */
    static hasLogin() {
        return App.cookie(App.getConfig('userCookieName')) && App.storage('username') && App.storage('pwd');
    }

    /**
     * 设置/获取/删除Cookie
     * @param name {String} 名称
     * @param value {String|undefined} 值
     * @param time {Number} 过期时间
     * @returns {null|String} 获取则返回值，设置或删除返回null
     */
    static cookie(name, value = undefined, time = 15 * 24 * 60 * 60 * 1000) {
        let result = null;
        if (value == undefined) {
            let matcher;    //  匹配成功返回的数组
            let cookies = plus.navigator.getCookie(App.getConfig('host') + "/");    //  应用所有的cookie
            // matcher = document.cookie.match(new RegExp('\(\?\:\^\|\(\?\:;\\s\)\)' + name + '=\(\[\^;\]\*\)'));
            matcher = cookies != null ? cookies.match(new RegExp('\(\?\:\^\|\(\?\:;\\s\)\)' + name + '=\(\[\^;\]\*\)')) : cookies;
            if (matcher != null) result = decodeURI(matcher[1]);
        }
        else {
            if (time === 0) {
                // plus.navigator.removeCookie();  //  android不支持只删除单一cookie
                plus.navigator.removeAllCookie();   //  清除应用所有cookie
            }
            else {
                let exp = new Date()
                    , cookieValue;
                exp.setTime(exp.getTime() + time);
                cookieValue = name + "=" + value + "; expires=" + exp.toGMTString() + "; path=/";
                plus.navigator.setCookie(App.getConfig('host') + '/', cookieValue);
                if (App.getConfig("environment") === "dev") {
                    plus.navigator.setCookie(App.getConfig("devHost") + '/', cookieValue);
                }
            }
        }
        return result;
    }

    /**
     * 通过封装storage进行存/取/清空
     * @param {String} key 键
     * @param {String,optional} value 值
     */
    static storage(key, value) {
        let ls = window.localStorage || plus.storage;
        if (value == undefined) {
            return ls.getItem(key);
        }
        if (value === '') {
            ls.removeItem(key);
            return;
        }
        ls.setItem(key, value);
    }

    /**
     * 下载封装
     * @param {String} url -下载地址
     * @param {Object} options -选项 [{String} successText -成功提示文字,{String} errorText - 错误提示文字,{Function} successCB - 成功回调,{Function} errorCB -错误回调]
     * @param {Object} downLoadOptions - 下载选项
     */
    static downLoad(url = App.getConfig('host') + "/app/update.wgt", options, downLoadOptions = {}) {
        options.successText = options.successText || "下载完成";
        options.errorText = options.errorText || "下载失败";
        plus.downloader.createDownload(url, downLoadOptions, function (download, status) {
            if (status == 200) {
                plus.nativeUI.toast(options.successText);
                if (options.successCB) options.successCB(download);
            } else {
                plus.nativeUI.alert(options.errorText);
                if (options.errorCB) options.errorCB(download, status);
            }
        }).start();
    }

    /**
     * 安装文件
     * @param {String} path 文件路径
     * @param {Object} options -选项 [{String} successText -成功提示文字,{String} errorText - 错误提示文字,{Function} successCB - 成功回调,{Function} errorCB -错误回调]
     * @param {Object} installOptions - 安装选项
     */
    static installPackage(path, options = {}, installOptions = {}) {
        options.successText = options.successText || "正在安装";
        options.errorText = options.errorText || "安装失败";
        plus.nativeUI.toast(options.successText);
        plus.runtime.install(path, installOptions, function () {
            plus.nativeUI.toast(options.successText);
            if (typeof options.successCB === 'function') options.successCB();
        }, function () {
            plus.nativeUI.alert(options.errorText);
            if (typeof options.errorCB === 'function') options.errorCB();
        });
    }
}

export {App};