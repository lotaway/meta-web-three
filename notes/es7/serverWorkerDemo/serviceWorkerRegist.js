// http://web.jobbole.com/92659/,目前Chrome/Firefox/Opera支持serviceWork，Safari和Edge正准备支持

window.addEventListener("load", () => {
    /*
    注册有且只有唯一一个的Service Workder。
    注册后会启动后台进程监听同域名的所有页面。
    `sw.config.v3.js`是运行环境文件，该文件的路径会决定sw的管理范围，如/page/sw.js只能管理/page路径下的页面和资源，无法处理/api路径下的，因此一般注册到顶级目录。
    刷新页面不会导致重新注册，也不会再重新触发install和active事件，更新配置文件会导致重新注册。
    重新注册后会先触发install和active并进入waiting状态，等待重启浏览器后才会生效。通过install里调用skipWaiting可以跳过等待。
    */
    if (!"serviceWorker" in navigator) {
        return console.log("不支持serviceWorker");
    }
    //  注册
    navigator.serviceWorker.register("/sw.config.v3.js")
        .then(reg => {
            console.log("注册成功：" + reg);
        })
        .catch(error => {
            console.log("注册失败：" + error);
        });

    //  订阅消息
    navigator.serviceWorker.addEventListener("message", event => {
        const MSG = event.data;

        switch (MSG.type) {
            case "pageUpdate":
                if (window.location.href === MSG.url) {
                    console.log("收到消息，这个页面已更新");
                    // window.location.reload();    //  已在sw.config.v3.js文件中进行新页面内容的返回，无法重新加载
                }
                break;
            default:
                console.log("不支持的消息类型");
                break;
        }
    });

    window.addEventListener("something happen", function () {
        //  向serviceWorker推送消息
        navigator.serviceWorker.controller.postMessage({
            type: "deleteCache",
            desc: "页面内容被用户更改了，删除页面缓存，下次读取新内容",
            url: window.location.href
        });
    })

});