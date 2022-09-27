// serviceWorker运行环境文件，上下文中无法获取到window、document等对象，只能使用部分内置方法，如fetch、Caches、Request、Response、Url等
(() => {
    const CACHE_NAME = "fed-cache"; //  缓存库名称
    //  内部封装方法
    let util = {
        //  请求（此文件内的请求不会再被active事件里的fetch拦截）
        fetch(request, callback = () => {
        }) {
            return fetch(request).then(response => {
                // 跨域的资源不能缓存（跨域时response.status==0）
                if (!response || response.status !== 200 || response.type !== "basic") {
                    return response;
                }
                util.saveCache(request, response.clone());
                callback();

                return response;
            });
        },
        //  判断存入缓存，缓存级别 sw>Application Cache manifest>http
        saveCache(request, response) {
            //    后台页面、preview链接不缓存，非GET请求无法缓存（可以手动修改method进行缓存）
            if (request.method === "GET" && request.url.indexOf("/backstage") === -1 && request.url.indexOf("preview_id") === -1) {
                caches.open(CACHE_NAME).then(cache => {
                    cache.put(request, response);
                });
            }
        },
        //  删除缓存
        deleteCache(url) {
            caches.open(CACHE_NAME).then(cache => {
                console.log("删除缓存：" + url);
                // todo 是否需要立即通知页面更新
                return cache.delete(url, {
                    ignoreVary: true
                });
            })
        },
        //  页面更新事件
        _pageUpdateTime: {},
        //  判断更新页面
        pageNeedUpdate(request) {
            const url = new URL(request.url),
                PAGE_NAME = util.getPageName(url);
            let jsonRequest = new Request("/cache/" + PAGE_NAME + "config.json");

            console.log("check page update time");

            //  获取页面一一对应的配置文件中的更新时间，确认缓存是否需要更新
            return util.fetch(jsonRequest).then(response => response.json().then(content => util._pageUpdateTime[PAGE_NAME] !== content.updateTime)).catch(() => false);
        },
        //  更新页面，并进行一些处理
        updatePage(request) {
            const url = new URL(request.url);

            return util.fetch(request).then(response => {
                //  推送更新消息
                util.postMessage({type: "pageUpdate", desc: "页面即将进行更新", url: url.href});

                return response;
            })
        },
        //  获取页面唯一名称
        getPageName(url) {
            return url.pathname;
        },
        postMessage(content) {
            clients.matchAll().then(allClients => {
                allClients.forEach(client => client.postMessage(content));
            });
        }
    };
    //  提供给外部的消息处理方法
    const messageEvent = {
        deleteCache: util.deleteCache
    };

    //  监听sw安装事件
    this.addEventListener("install", event => {
        this.skipWaiting(); //  跳过等待，立即生效。否则首次安装和更新时预加载后会等待到下次启动时才生效
        console.log("install service worker");
        // 创建和打开一个缓存库
        caches.open(CACHE_NAME);
        // 要缓存的地址
        let cacheResources = [
            /*首页*/"https://fed.renren.com/?launcher=true"
        ];
        event.waitUntil(
            caches.open(CACHE_NAME).then(cache =>
                //    请求资源并添加到缓存里面去
                cache.addAll(cacheResources))
        );
    });
    //  安装完成后会触发激活事件，日常操作都在此事件中进行
    this.addEventListener("active", e => {
        console.log("service worker is active");
        //  拦截fetch事件
        this.addEventListener("fetch", event => {
            event.respondWith(
                //  判断是否命中缓存在（url和header都一致才算相同资源，可以设置第二个参数Object {Boolean ignoreVary}忽略header）
                caches.match(event.request)
                    .then(response => {
                        //  命中直接返回
                        if (response) {
                            //  若是html文件，需要确认是否已更新过，防止更改sw文件版本时无法及时获取
                            if (response.headers.get("Content-Type").indexOf("text/html") > -1) {
                                return util.pageNeedUpdate(event.request.clone()).then(pageNeedUpdate => pageNeedUpdate
                                    ? util.updatePage(event.request)
                                    : response);
                            }

                            return response;
                        }

                        return util.fetch(event.request.clone());
                    })
            );
        });
        //  订阅消息
        this.addEventListener("message", event => {
            const MSG = event.data;

            console.log("sw收到消息：" + MSG);
            if (typeof messageEvent[MSG.type] === "function") {
                messageEvent[MSG.type](MSG.url);
            }
        })
    });
})();