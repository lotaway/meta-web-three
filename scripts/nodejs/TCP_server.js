/**
 * TCP服务，大多用于长连接类，借助net模块可以轻松创建tcp服务
 * 使用telnet进行本地测试
 */
var net = require('net');

var app = net.createServer(function (socket) {

    console.log("net server connected");

    socket.on('data', function (data) {
        //  接收的data是Buffer，可以直接用toString转化为字符串
        console.log('server get data: ' + data + 'form ' + /*连接的客户端地址*/socket.remoteAddress + ':' + /*连接的客户端端口*/socket.remotePort);
    });

    socket.on('close',function(data){
        console.log('server disconnected ');
    });

});

app.listen('10101','127.0.0.1');


//使用net模块也可以创建tcp客户端
var net = require('net');
var tcpClient = net.Socket();

tcpClient.connect('10101','127.0.0.1', function () {

    console.log('net client connect success');

    tcpClient.write('client send some msg');

});

tcpClient.on('data', function (data) {
    //  接收的data是Buffer，可以直接用toString转化为字符串
    console.log('client get data:' + data.toString());
});