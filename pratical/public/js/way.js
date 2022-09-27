window.way = (function () {
    /**
     * 检查元素是否拥有样式名
     * @param c 样式名
     * @returns {boolean}
     */
    Element.prototype.hasClass = function (c) {
        return new RegExp("(^|\\s)" + c + "(\\s|$)").test(this.className);
    };
    /**
     * 把样式名赋给元素
     * @param c 样式名
     */
    Element.prototype.addClass = function (c) {
        return !this.hasClass(c) && (this.className += " " + c);
    };
    /**
     * 从元素中移除样式名
     * @param c 样式名
     */
    Element.prototype.removeClass = function (c) {
        return this.hasClass(c) && (this.className = this.className.replace(new RegExp("(^|\\s)" + c + "(\\s|$)"), ""));
    };

    /*if (!window.addEventListener) {
        //  兼容监听【添加/移除】方式
        HTMLDocument.prototype.addEventListener = function (event, fn, args, isCatch) {
            var t = this;
            isCatch = isCatch || false;
            args = args || [];
            
            //if (window.addEventListener) t.addEventListener(e, f.apply(t, p), c);
            //else
            window.attachEvent ? t.attachEvent("on" + event, function () {
                fn.call(t, args)
            }, isCatch) : t.setAttribute("on" + event, fn);
        };
        HTMLDocument.prototype.removeEventListener = function (event, fn, c) {
            var t = this;
            c = c || false;
            //if (window.addEventListener) t.removeEventListener(e, f, c);
            //else
            window.detachEvent ? t.detachEvent(event, fn, c) : t.removeAttribute("on" + event);
        };
        //  补充数组查找字符的方法
        Array.prototype.indexOf = function (e, i) {
            var l = this.length >>> 0;
            i = Number(i) || 0;
            i = (i < 0) ? Math.ceil(i) : Math.floor(i);
            if (i < 0) i += l;
            for (; i < l; i++) {
                if (i in this && this[i] === e) return i;
            }
            return -1
        };
        //  补充用样式名获取dom节点的方法
        HTMLDocument.prototype.getElementsByClassName = function (className, element) {
            var children = (element || this || document).getElementsByTagName("*");
            var elements = [];
            for (var i = 0; i < children.length; i++) {
                if (children[i].hasClass(className)) elements.push(children[i])
            }
            return elements
        };
        //  补充占位文字
        /!*var a = document.getElementsByTagName("input").concat(document.getElementsByTagName('textarea'));
         for (var i = 0, l = a.length; i < l; i++) {
         var holder = a[i].getAttribute("placeholder");
         if (holder && a[i].value == "") a[i].value = holder
         }*!/
    }*/

    var way = {
        //  获取字符串中的数字
        getStringNum: function (str) {
            return str.replace(/[^\d]/ig, "");
        },
        //  简易轮播
        banner: function (slide) {
            var t = this;

            function init() {
                slide = slide || "swiper-slide";
                t.slide = document.getElementsByClassName(slide);
                if (t.slide.length <= 1) return;
                for (var i = 0, l = t.slide.length; i < l; i++) {
                    t.slide[i].style.cssText = "display:none;filter:alpha(opacity=0);z-index:0;"
                }
                t.slide[0].style.cssText = "display:block;filter:alpha(opacity=100);z-index:1;";
                setInterval(t.play, 6000);
            }

            t.play = function () {
                for (var i = 0, l = t.slide.length; i < l; i++) {
                    if (t.slide[i].style.display == "block") {
                        t.slide[i].style.cssText = "display:none;z-index:0;";
                        var n = i < l - 1 ? i + 1 : 0;
                        t.slide[n].style.cssText = "display:block;filter:alpha(opacity=20);z-index:1;";
                        t.clock = setInterval("t.anim(" + n + ")", 64);
                        break;
                    }
                }
            };
            //  todo 轮播动画
            t.anim = function (n) {
                var pass = parseInt(way.getStringNum(t.slide[n].style.filter)) + 10;
                if (pass > 100) {
                    pass = 100;
                    clearTimeout(t.clock)
                }
                t.slide[n].style.filter = "alpha(opacity=" + pass + ")"
            };
            init();
        },
        //  检查浏览器
        browser: {
            check: function () {
                var ua,
                    browser,
                    version,
                    trim_Version,
                    Sys = {};

                ua = navigator.userAgent.toLowerCase();

                if (window.ActiveXObject) {

                    Sys.ie = ua.match(/msie ([\d.]+)/)[1];
                    if (!Sys.ie) return null;

                    browser = navigator.appName;
                    version = navigator.appVersion.split(";");
                    trim_Version = version[1].replace(/[ ]/g, "");

                    if (browser == "Microsoft Internet Explorer") {
                        switch (trim_Version) {
                            case "MSIE6.0":
                                return "ie6";
                                break;
                            case "MSIE7.0":
                                return "ie7";
                                break;
                            case "MSIE8.0":
                                return "ie8";
                                break;
                            case "MSIE9.0":
                                return "ie9";
                                break;
                            default:
                                return "ie5"
                        }
                    }
                    else {
                        return false
                    }
                }
            }
        },
        //  按需加载图片
        LazyLoader: function (target, animation) {

            var t = this;
            var array;

            t.array = [];

            if (target) {
                if (/^#/.test(target)) {
                    array = [];
                    array[0] = document.getElementById(target.split("#")[1]);
                }
                else if (/^\./.test(target)) {
                    array = document.getElementsByClassName(target.split(".")[1]);
                }
                else return;
                for (var i = 0; i < array.length; i++) {
                    t.array.push(array[i]);
                }
            }
            else return;

            t.timer = undefined;
            t.func = {
                init: function () {
                    t.func.check();
                    window.addEventListener('scroll', t.func.check);
                },
                check: function () {
                    if (!t.array.length) return;
                    if (t.timer) clearTimeout(t.timer);
                    t.timer = setTimeout(function () {
                        if (t.array.length) {
                            while (t.array.length) {
                                var ta = t.array[0],
                                    top = t.array[0].offsetTop;
                                while (ta.parentNode && ta.parentNode.tagName != "BODY") {
                                    ta = ta.parentNode;
                                    top += ta.offsetTop;
                                }
                                if (top < document.body.scrollTop + document.body.clientHeight) {
                                    t.func.loader(t.array.shift());
                                }
                                else break;
                            }
                        }
                    }, 40);
                },
                loader: function (e) {
                    var a = e || t.array[0],
                        img = new Image();
                    img.onload = function () {
                        setTimeout(function () {
                            var a = animation || 'anim-fade-in';
                            a.className += " " + a;
                            a.src = img.src;
                        }, 1000);
                    };
                    img.error = function () {
                        t.func.loader(e);
                    };
                    img.src = a.getAttribute('data-src');
                }
            };
            t.func.init();
        }
    };

    return way;

})();