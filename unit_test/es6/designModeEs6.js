/**
 * Created by lotaway on 2016/8/12.
 * https://segmentfault.com/a/1190000002800554
 */
'use strict';
+function () {

    // 单例模式   重点在于利用立即执行函数保存私有变量
    let singleMode = (function () {
        //  声明时使用括号强制运行了此函数，创建了私有变量instance
        var instance;
        return function (newInstance, proxy) {
            //  每次调用singleMode()实际运行的函数，可以获取到instance的值
            //  桥接模式  抽象与实现分离，令各自可以独立变化（这里体现为单例抽象和代理实现）
            if (proxy) instance = proxy.apply(this, newInstance);
            if (newInstance) instance = newInstance;
            return instance;
        };
    })();

    class Caller {
        constructor(words, proxy) {
            if (singleMode()) return singleMode(); // 首次调用singleMode并且没有传参时，instance仍为undefined，判断为false，所以不会此判断内的代码（返回singleMode()的返回值）
            this.foo = `first time to here and sand ${words}`;
            singleMode(this, proxy);
        }
    }

    var caller1 = new Caller('caller1');    //  caller1参数成功被赋值给foo
    var caller2 = new Caller('caller2');    //  由于单例不是初次传值，所以没有赋值，foo仍为caller1
    console.log(`caller1's foo: ${caller1.foo} and caller2's foo: ${caller2.foo}`);

    //  迭代器
    let agg = {
        data: [1, 2, 3, 4],
        //  直接使用[Symbol.iterator]指定对象的迭代方法（for...of)会调用[Symbol.iterator].next()方法进行遍历
        [Symbol.iterator](){
            let index = 0;
            return {
                next: () => this.hasNext() ? {value: this.data[index++], done: false} : {
                    value: undefined,
                    done: true
                },
                current: ()=> {
                    index--;
                    return this.next();
                },
                hasNext: ()=> index < this.data.length,
                rewind: ()=> index = 0
            }
        },
        //  使用生成器简易实现（生成器方法内部采用yield实现了next()方法
        [Symbol.iterator]: function*() {
            for (var i = 0; i <= this.data.length; i++) {
                yield i;
            }
        }
    };

    let iter = agg[Symbol.iterator]();
    console.log(iter.next());  // { value: 1, done: false }
    console.log(iter.next()); // { value: 2, done: false }
    console.log(iter.current());// { value: 2, done: false }
    console.log(iter.hasNext());// true
    console.log(iter.rewind()); // rewind!
    console.log(iter.next()); // { value: 1, done: false }
    // for...of方法会调用[Symbol.iterator].next()进行遍历
    for (let ele of agg) {
        console.log(ele);
    }


//  简单工厂模式，集合了多个类，因实例化类型在编译期无法确定。在实例化时才决定到底是创建xhr的实例, 还是jsonp的实例. 是由实例化决定的
    class Ajax {
        constructor(options) {
            var p, xhr;
            this.o = {
                url: "",
                type: "POST",
                data: "",
                dataType: "json",
                async: false,
                success: "",
                error: ""
            };
            if (typeof options === "object")
                for (p in this.o) {
                    if (options[p] != undefined) this.o[p] = options[p];
                }
            switch (this.o.dataType.toUpperCase()) {
                case "JSON":
                    this.o.dataType = "application/json";
                    break;
                case "JSONP":
                    this.o.dataType = "text/javascript";
                    break;
                case "TEXT":
                    this.o.dataType = "html/text";
                    break;
                case "XML":
                    this.o.dataType = "html/xml";
                    break;
                default:
                    break;
            }
            xhr = (function () {
                if (typeof XMLHttpRequest == "undefined") {
                    XMLHttpRequest = function () {
                        try {
                            return new ActiveXObject("Msxml2.XMLHTTP.6.0");
                        }
                        catch (e) {
                        }
                        try {
                            return new ActiveXObject("Msxml2.XMLHTTP.3.0");
                        }
                        catch (e) {
                        }
                        try {
                            return new ActiveXObject("Msxml2.XMLHTTP");
                        }
                        catch (e) {
                        }
                    };
                }
                return new XMLHttpRequest();
            })();
            return xhr;
        }

        done(fn) {
            if (typeof fn === 'function') {
                this.o.success = fn;
            }
        }

        onreadystatechange() {
            if (this.o.type.toUpperCase() != "JSONP") {
                //0表示未初始化，1表示正在加载，2表示加载完毕，3表示正在交互，4表示完成
                if (this.readyState == 4) {
                    var t;
                    switch (this.o.dataType) {
                        //访问返回的数据通过两个属性完成，一个是responseText，用于保存文本字符串形式的数据。另一个是responseXML，用于保存Content-Type头部中指定为text/xml的数据，其实是一个DocumentFragment对象
                        case "application/json":
                            if (typeof this.responseText === "string")
                                t = JSON.parse(this.responseText);
                            t = this.responseText;
                            break;
                        case "html/text":
                            t = this.responseText;
                            break;
                        case "html/xml":
                            t = this.responseXML;
                            break;
                        default:
                            break;
                    }
                    //200表示数据全部接受完毕
                    if (this.status == 200) this.o.success(t);
                    else this.o.error(t);
                }
            }
        }

        start() {
            if (this.o.type.toUpperCase() == "JSONP") {
                var script = document.createElement('script');
                script.setAttribute('src', this.o.url + "?" + this.o.data + '&callback=' + this.o.success);
                document.getElementsByTagName('head')[0].appendChild(script);
            }
            else {
                this.open(this.o.type, this.o.url, this.o.async);
                //xmlHttpRequest.charset = "UTF-8";
                if (this.o.type.toLocaleUpperCase() == "POST") {
                    this.setRequestHeader("Content-Type", this.o.dataType);
                }
                this.send(this.o.data);
            }
        };
    }
//  实例化为json类型
    let ajaxRequest1 = Ajax({url: 'your_ajax_url', type: 'post'});
    ajaxRequest1.start();
    ajaxRequest1.done(fn);
//  实例化为jsonp类型
    let ajaxRequest2 = Ajax({url: 'your_ajax_url', type: 'jsonp'});
    ajaxRequest2.start();
    ajaxRequest2.done(fn);


    //订阅/发布模式 被Node原生的Events模块所支持，同样结合默认参数，for...of遍历等特性，代码的减少以及可读性的增加都是可观的：
    class Events {
        constructor() {
            this.subscribers = new Map([['any', []]]);
        }

        on(fn, type = 'any') {
            let subs = this.subscribers;
            if (!subs.get(type)) return subs.set(type, [fn]);
            subs.set(type, (subs.get(type).push(fn)));
        }

        emit(content, type = 'any') {
            for (let fn of this.subscribers.get(type)) {
                fn(content);
            }
        }
    }

    let event = new Events();

    event.on((content) => console.log(`get published content: ${content}`), 'myEvent');
    event.emit('jaja', 'myEvent'); //get published content: jaja


}();