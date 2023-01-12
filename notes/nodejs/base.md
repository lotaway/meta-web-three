@[TOC](Nodejs学习笔记 - 基础模块)

# 导入模块

以下方式将从node_modules文件夹里查找文件，若找不到将从nodejs内置模块查找文件，若还是找不到将从当前文件相对位置查找合适文件
const http = require("http")

## 也可以用es6方式导入

import http from "http"

# 创建服务器

## 显式创建

直接实例方式创建，一般不用

```javascript
const http = require("http")
const server = new http.Server()
server.on('request', (req, res) => {
    //  处理请求和响应，详见下一种写法
});
```

## 封装创建

通过生产器内部创建，纯nodejs常用写法，但企业中一般是在express或koa等框架基础上通过框架语法创建。

```javascript
const http = require("http")
const qs = require('querystring')
const server2 = http.createServer((req, res) => {
    let data = ''
    request.on('data', chunk => {
        data += chunk
    })
    request.on('end', () => {
        data = qs.parse(data)
    })
    response.writeHead(200, {'Content-Type': 'text/html'});
    response.write('<span class="response">This is the response content.</span>');
    response.end('<p class="end">It\'s end.</p>');   //结束发送
});
server2.listen(3000)
console.log("HTTP server is listening at port 3000")
```

### ServerRequest事件

* data：当请求体数据到来时，该事件被触发。该事件提供一个参数chunk，表示接收到的数据。该事件可能会被调用多次。
* end：当请求体数据传输完成时，该事件被触发，此后将不会再有数据到来。
* close： 用户当前请求结束时，该事件被触发。不同于 end，如果用户强制终止了传输，也还是调用close。

### ServerRequest属性

* complete 客户端请求是否已经发送完成
* httpVersion HTTP协议版本，通常是1.0或1.1
* method HTTP 请求方法，如GET、POST、PUT、DELETE等url原始的请求路径，例如/static/image/x.jpg 或/user?name=byvoid
* headers HTTP 请求头
* trailers HTTP 请求尾（不常见）
* connection 当前HTTP连接套接字，为net.Socket的实例
* socket connection属性的别名
* client client 属性的别名

## express创建

```javascript
const express = require("express")
const app = express()
app.use("/", (req, res) => {
    //  处理请求
})
app.listen(3000)
```

## koa创建

```javascript
const koa = require("koa")
const app = new Koa()
app.use((context, next) => {
    //  处理请求
})
app.listen(3000)
```

# 创建https服务

使用https模块创建安全的网络响应，需提供SSL证书

```javascript
const https = require('https')
const fs = require('fs')
var options = {
    key: fs.readFileSync('ssh_key.pem'),    //  密钥
    cert: fs.readFileSync('ssh_cert.pem')   //  证书
}
https.createServer(options, (request, response) => {
    //  处理请求
})
```

# 发送请求

如果是简单的接口请求，直接用fetch更好，如果需要详细专业配置才用以下方法

```javascript
const http = require('http')
const qs = require('querystring')
const content = qs.stringify({
    //发送的内容
    name: 'lotaway',
    email: '576696294@qq.com',
    address: 'shantou,China'
})
const reqOpt = {
    //  请求的选项
    //host: 'www.lotaway.com',    //请求网站的域名或IP地址
    path: '/api/post/test?keyword=1', //请求的相对路径和数据参数
    method: 'POST', //请求方法，默认为GET
    headers: {  //请求头
        'Content-Type': 'application/json',
        'Content-Length': content.length
    }
}
var req = http.request(reqOpt, res => {
        //  接收数据的回调函数
        res.setEncoding('utf8')
        res.on('data', data => {
            console.log(data)
        })
    }
)
req.write(content)
req.end()  //   结束请求
```

## GET请求的简便写法

```javascript
const http = require('http');
http3.get({host: 'www.lotaway.com'}, function (res) {
    res.setEncoding('utf8');
    res.on('data', function (data) {
        console.log(data);
    });
});
```

# 事件监听器

```javascript
const eventEmitter = new (require('events')).EventEmitter()
//  注册事件监听器，设置自定义事件和回调
eventEmitter.on('myEventName', function (arg1, arg2) {
    console.log("trigger the event with arg1:" + arg1 + "and arg2:" + arg2);
});
eventEmitter.once('only-trigger-once', function () {
    console.log("only trigger on the first time");
});
eventEmitter.on('error', function () {
    console.log("into the error event");
//    一般都要设置错误回调，否则错误时会退出程序并打印调用栈
});
//  用延时来模拟触发事件
setTimeout(function () {
    eventEmitter.emit('myEventName', 'A1', 'A2');
}, 2000);
// 移除指定事件的某个监听器，listener必须是该事件已经注册过的监听器。
//EventEmitter.removeListener(event, listener)
eventEmitter.removeAllListeners(); // 移除所有事件的所有监听器，参数可以指定event，则移除指定事件的所有监听器。
```

# 核心模块

核心模块由一些精简高效的库组成，为nodejs提供了基本的API，常用内容介绍：

* 1、全局对象
* 2、常用工具；
* 3、事件机制；
* 4、文件系统访问；
* 5、HTTP服务器与客户端

## 全局对象

相比js本身的全局对象是window，nodejs的全局对象是global

## process

* 用于描述当前nodejs进程状态
* process.argv是命令行参数数组，第一个元素是node，第二个是脚本文件名，第三个开始是运行参数。

```javascript
  console.log(process.argv)
//  ['node','/byvoid/argv.js','2016','==v','Carbo Kuo']
```

## 执行命令行

通过child_process模块的exec方法可在当前环境执行命令行命令，规则：`child_process.exec(command[, options][, callback])`，示例：

```javascript
const {exec} = require("child_process")
//  exec("ls")
//  ls是linux的输出当前文件夹下的目录，window要用dir
exec("dir", {
    cmd: "../"  //  在上一级路径执行命令
}, (err, stdout, stderr) => {
    //  执行过程遇到错误直接抛出
    if (err) {
        throw err
    }
    console.log(stdout) //  正常执行命令后的打印命令行返回内容
    console.error(stderr)   //  执行命令错误后打印命令行返回的错误信息
})
```

## stdout & stdin 命令行输出输入流

process.stdout是标准输出流，而process.stdin是标准输入流，但初始时是暂停的，要输入数据需要恢复流并配上回调函数：

```javascript
process.stdin.resume()
process.stdin.on('data', data => {
    process.stdout.write('read from console:' + data.toString())
})
```

## nextTick

为事件循环设置一项任务，把事件拆解成各小的部分，避免一个事件占用太多CPU时间，这比setTimeout(fn,0)的效率高得多

```javascript

function doSomething(next) {
    console.log("do something first")
    process.nextTick(next)
}

function somethingElse() {
    console.log("do something else")
}

doSomething(somethingElse)
```

还有precess.platform,process.pid,process.execPath,process.memoryUsage()等方法，以及POSIX进程信号响应机制

# 加密

加密最常见的就是将账户密码进行MD5加密存储，内置库提供了相应的方式：

```javascript
const crypto = require("crypto")
const md5 = crypto.createHash("md5")
md5.update("这里是要加密的字符串内容")
const cryptoContent = md5.digest("hex") //  加密好的内容，默认全小些
```

也有更简单的第三方模块（需要通过npm安装）：

```javascript
const md5 = require("md5")
const cryptoContent = md5("这里是要加密的字符串内容")   //  加密好的内容，默认全小写
```