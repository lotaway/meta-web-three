//     web socket 多人聊天室，本例子中会让game1客户端发送信息到game2客户端中

//  服务端
var websocketModule = require("nodejs-websocket");

console.log("开始建立连接....");

var game1 = null
    , game2 = null
    , game1Ready = false
    , game2Ready = false
    , game1Str = ""
;

var server = websocketModule.createServer(function (connect) {
    connect.on("text", function (str) {
        const strObj = JSON.parse(str);

        console.log("收到的信息是：" + str);
        if (strObj.id === "game1") {
            game1 = connect;
            game1Ready = true;
            game1Str = strObj.msg;
            connect.sendText("success");
        }
        if (strObj.id === "game2") {
            game2 = connect;
            game2Ready = true;
        }
        if (game1Ready && game2Ready) {
            game2.sendText(game1Str);
        }
    });

    connect.on("close", function (code, reason) {
        console.log("关闭链接");
    });

    connect.on("error", function (code, reason) {
        console.log("异常关闭");
    });
});

server.listen(8001);

//  客户端（game1）
var webSocket = new WebSocket('ws://localhost:8001');
webSocket.onopen = function (event) {
    console.log('open');

    // 发送一个初始化消息
    webSocket.send(JSON.stringify({id: 'game1', msg: "hello"}));

    // 监听消息
    webSocket.onmessage = function (event) {
        console.log('Client received a message', event);
    };

    // 监听Socket的关闭
    webSocket.onclose = function (event) {
        console.log('Client notified socket has closed', event);
    };

    webSocket.onerror = function () {
        console.log("连接出错");
    }

    // 关闭Socket....
    //socket.close()
};

//  相同方式创建game2客户端，就可以接受到消息。