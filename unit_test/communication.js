/**
 * Created by lw on 2016/5/17.
 */
//     web socket 多人聊天室
var wsImp = new WebSocket('ws://localhost:8080');
wsImp.onopen = function (event) {
    console.log('open');

    // 发送一个初始化消息
    wsImp.send('client listening?name=test');

    // 监听消息
    wsImp.onmessage = function(event) {
        console.log('Client received a message',event);
    };

    // 监听Socket的关闭
    wsImp.onclose = function(event) {
        console.log('Client notified socket has closed',event);
    };

    // 关闭Socket....
    //socket.close()
};