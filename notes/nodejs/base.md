@[TOC](Nodejs学习笔记 - 基础模块快速入门)

# 介绍

Nodejs相当于在特化的服务端环境下运行javascript。
失去了客户端特有的window、document等对象，换来了包括global对象和核心模块等各种服务端特化功能。
核心模块由一些Nodejs内置的库组成，为nodejs提供了众多功能，如：

* HTTP服务器
* 文件管理
* 进程管理
* 事件机制
* 常用工具

# 使用核心模块

虽然Nodejs提供了特化功能，但不同于js可以直接调用，Nodejs提供的功能都需要通过导入模块才能使用，可以理解为像java、.net导入包的概念一样。

```javascript
const http = require("http")
```

## 也可以用es6方式导入

```javascript
import http from "http"
```

以上方式将从node_modules文件夹里查找符合条件的第三方库（类似SDK概念），若找不到才会去查找Nodejs核心模块，若还是找不到将从当前文件相对位置查找符合要求的文件。

# 创建网络服务

可以当做创建一个持续运行的网站服务实例，用户可以通过访问网址（HTTP协议）访问到这个服务，而服务可以通过请求和响应进行交互。

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

通过生产器内部创建，纯nodejs的常用写法。

```javascript
const http = require("http")
const qs = require('querystring')
const server = http.createServer((req, res) => {
    let data = ''
    //  每次接收到数据就会触发，要注意数据并不是一次性传输完成的
    request.on('data', chunk => {
        data += chunk
    })
    //  数据传输完成后触发
    request.on('end', () => {
        data = qs.parse(data)
    })
    //  请求结束时，无论是数据传输完成还是强制中止都会触发
    request.on("close", () => {

    })
    response.writeHead(200, {'Content-Type': 'text/html'});
    response.write('<span class="response">This is the response content.</span>');
    response.end('<p class="end">It\'s end.</p>');   //结束发送
});
server.listen(3000)
console.log("HTTP server is listening at port 3000")
```

以上是nodejs里创建服务的方式，但企业一般在使用express或koa等框架的基础上，通过框架语法创建。

## express框架下创建服务

```javascript
const express = require("express")
const app = express()
app.use("/", (req, res) => {
    //  处理请求
})
app.listen(3000)
```

## koa框架下创建服务

```javascript
const koa = require("koa")
const app = new Koa()
app.use((context, next) => {
    //  处理请求
})
app.listen(3000)
```

大部分框架会集成创建服务、路由、后端模板、中间件等功能，感兴趣的可以去查看express和koa框架教程文档，这里只简单介绍，不再展开。

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

类似于js的fetch方法，但Nodejs里没有fetch，只能通过http或者request模块发送请求。

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
const http = require('http')
http.get({host: 'www.lotaway.com'}, function (res) {
    res.setEncoding('utf8');
    res.on('data', function (data) {
        console.log(data);
    });
})
```

# events 事件监听器

```javascript
const events = require('events')
const eventEmitter = new events.EventEmitter()
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

# Buffer 流

`Buffer`既一种二进制的流，大部分数据传输时实际就是作为`Buffer`存在，包括但不限于HTTP上传下载文件、视音频播放、本地文件读取与写入等。
掌握`Buffer`和流的使用，充实基础才能解决实际面临的数据问题

## 字符串转换成Buffer

通过`Buffer.from(string)`的方式可以将内容变成Buffer格式
参数：源Buf（必须），偏移量，长度，编码（默认utf8）

```javascript
var strBuffer = Buffer.from("something")
console.log(strBuffer.length)   //  和数组一样通过length获取长度
```

## 指定Buffer长度

可以通过`Buffer.alloc(size)`指定固定长度，当写入字符串内容时会丢弃多余字符

```javascript
var buf2 = Buffer.alloc(8)
buf2.write('something')
console.log(buf2.length)
//  获取到的长度固定是8
```

## 数组转换成Buffer

更常用的是写入数组`Buffer.from(array)`，可以直接通过下标调用

```javascript
var buf3 = Buffer.from([0, 1, 2, 3])
console.log(buf3[0])
//  获取到的内容为原数组索引0位置的内容：0
```

## 复制Buffer

复制是指从一个buffer实例复制到另外一个实例，需要的参数是：源buffer（必须），目标写入开始位置，源读取开始位置，源结束位置

```javascript
let originBuffer = Buffer.from("这就是要复制的内容")
let targetBuffer = Buffer.allocUnsafe(originBuffer.length)
targetBuffer.copy(originBuffer, 0, 0, originBuffer.length)
```

这种复制适合两个Buffer实例在不同位置创建的

## 更直接的复制方式

这种适合在同个位置或函数内进行复制

```javascript
let originBuffer = Buffer.from("这就是要复制的内容")
let targetBuffer = Buffer.from(originBuffer)
```

## 更多Buffer方法

```javascript
Buffer.toString(buffer, encoding, start = 0, end = buffer.length)
buffer.slice(start, end)
Buffer.compare(oneBuffer, otherBuffer)
buffer.equal(otherBuffer)
buffer.fill(value, offset, end)
```

# 文件操作 fs

fs模块是文件操作的封装，提供了文件读取、写入、更名、删除、遍历目录、链接等POSIX操作，并且都有同步和异步方法。

## 异步读取文件

```javascript
const fs = require('fs')
//  读取文件
fs.readFile('./logo.png', (err, fileBuffer) => {
    //  读取出的内容将是`Buffer`类型
    console.log(Buffer.isBuffer(fileBuffer))
    //  写入文件
    fs.writeFile('logo-2.png', fileBuffer, err => {
        if (err) {
            //  没有错误时err为nll或undefined，有错误发生时为Error对象实例
            console.error(err)
            return
        }
        console.log("写入完成")
    })

    //  将`Buffer`转换成常人看的字符串内容
    var fileStr = fileBuffer.toString('base64')
    console.log(fileStr)

    //  将字符串转回为`Buffer`，内容和原本的Buffer是一致的
    var fileBuffer2 = new Buffer(fileStr, 'base64')
    console.log(Buffer.compare(fileBuffer, fileBuffer2))
    fs.writeFile('logo-decoded.png', fileBuffer2, err => {
        if (err) console.log(err)
        else console.log("文件写入完成")
    })
});
```

## 同步读取文件

该读取会持续等待直到文件内容返回

```javascript
const fs = require("fs")
const fileBuffer = fs.readFileSync('aFile.txt', 'utf-8')
console.log(`文件读出的内容：${fileBuffer}`);
```

## 异步写入文件

可选覆盖 / 追加、自动创建路径文件夹等

```javascript
const fs = require("fs")
var writeData = "要写入的新内容";
fs.writeFile("./path/to/newFile.txt", writeData, {
    encoding: "utf8",
    flag: "w+"  //  文件不存在则自动创建
}, err => {
    if (err) {
        console.error(err)
        return
    }
    console.log("写入文件成功")
})
```

## 直接打开文件

直接打开文件，就像你用鼠标双击打开文件一样，此时将通过手动管理所有读写权限请求、缓冲区、文件指针等（readFile/writeFile是封装好的方法）
接受两个必选参数，path 为文件的路径，flags可以是以下值。

* r ：以读取模式打开文件。
* r+ ：以读写模式打开文件。
* w ：以写入模式打开文件，如果文件不存在则创建。
* w+ ：以读写模式打开文件，如果文件不存在则创建。
* a ：以追加模式打开文件，如果文件不存在则创建。
* a+ ：以读取追加模式打开文件，如果文件不存在则创建。
* mode 参数用于创建文件时给文件指定权限，默认是0666。
* 回调函数将会传递一个文件描述符fd。
* 1、文件权限指的是POSIX操作系统中对文件读取和访问权限的规范，通常用一个八进制数来表示。例如0754表示文件所有者的权限是7（读、写、执行），同组的用户权限是5（读、执行），其他用户的权限是4（读），写成字符表示就是
  -rwxr-xr--。
* 2、文件描述符是一个非负整数，表示操作系统内核为当前进程所维护的打开文件的记录表索引。

```javascript
const fs = require("fs")
fs.open('aFile.txt', 'r', (err, fd) => {
    if (err) {
        console.error(err);
        return;
    }
    var buf = new Buffer(8);
    fs.read(fd, buf, 0, 8, null, (err, bytesRead, buffer) => {
        if (err) {
            console.error(err)
            return
        }
        console.log('bytesRead:' + bytesRead);
        console.log(buffer);
        /*运行结果则是：
         bytesRead: 8
         <Buffer 54 65 78 74 20 e6 96 87>*/
        //  完成后（对于打开文件的操作）需要关闭文件
        fs.close(fd, err => {
            if (err) {
                console.error(err);
            } else {
                console.log("关闭文件成功");
            }
        });
    });
});
```

## 流输出

使用流输出，可断续

```javascript
const fs = require('fs');
var readStream = fs.readFileSync('./logo.png');
// 数据流是分段传输的，需要通过回调重新拼接成完整`Buffer`
let fileBuffers = []
readStream
    //  获取分段流
    .on('data', fileBuffer => {
        console.log(Buffer.isBuffer(fileBuffer))
        fileBuffers.push(fileBuffer)
    })
    //  可读
    .on('readable', () => {
        console.log('data readable')
    })
    //  数据传输完成
    .on('end', () => {
        console.log('data end')
    })
    //  异常
    .on('error', err => {
        console.log(err)
    })
    //  流传输关闭
    .on('close', () => {
        console.log('steam close')
    })
//  流暂停与继续
readStream.pause()
setTimeout(() => {
    readStream.resume()
}, 1000)
```

## 输入输出流

这种方式没有区分是读取文件还是直接读取Buffer，都统一处理成流，因此读取文件时不是直接获取到整个文件内容，而是分段读取。
写入流也是相同情况，写入文件时并不是一次性全部写入，但是按照分段流一段一段写入。

```javascript
const fs = require("fs")
var audioReadStream = fs.createReadStream('./audio/1.mp4')
var targetWriteStream = fs.createWriteStream('./audio/1-copy.mp4')
// readStream.setEncoding("utf8")   //  文件类需要定好编码
audioReadStream.on('data', chunk => {
    // 执行写入
    if (targetWriteStream.write(chunk) === false) {
        console.log('输入流未完成缓存，需等待drain事件')
        audioReadStream.pause()
    }
    // targetWriteStream.write(chunk, "utf8")   //  文件类需要加编码
})
    .on('end', () => {
        console.log("输出流结束时把输入流也关闭掉")
        targetWriteStream.end()
    })
targetWriteStream
    .on('drain', () => {
        console.log('输入流完成缓存写入后触发事件，恢复输出流')
        audioReadStream.resume()
    })
    .on("finish", () => {
        console.log("写入完成回调")
    })
    .on("error", err => {
        console.log(er.stack)
    })
```

也可以等待输出流将文件读取完毕后再一次性用输入流写入，但这样和直接写入文件没有区别

## 管道形式传输流信息

```javascript
var readerStream2 = fs.createReadStream("aFile.txt")
var writerStream2 = fs.createWriteStream("fileCopy.txt")
readerStream2.pipe(writerStream2)
console.log("管道读写操作执行完成")
```

## 压缩/解压也可使用管道流形式

```javascript
const fs = require("fs")
const zlib = require("zlib")
// 使用zlib库将文件压缩成gz格式
fs.createReadStream("aFile.txt").pipe(zlib.createGzip()).pipe(fs4.createWriteStream("aFile.txt.gz"))
// 使用zlib库解压文件
fs.createReadStream("aFile.txt.gz").pipe(zlib.createGunzip()).pipe(fs.createWriteStream("aFile.txt"))
```

# 路径操作 path

path的模块，可以帮你标准化，连接，解析路径，从绝对路径转换到相对路径，从路径中提取各部分信息，检测文件是否存在。总的来说，path模块其实只是些字符串处理，而且也不会到文件系统去做验证（path.exists函数例外）。

## 路径标准化 normalize

将用户输入或者配置文件中获取到的路径进行一次标准化，可以将“..”、“.”、“//”等特定位置或错误的情况进行过滤重整后输出，如：

```javascript
const {normalize} = require('path')
normalize('/foo/bar//baz/asdf/quux/..'); // 将双划线和最后的..处理掉，变成：/foo/bar/baz/asdf
```

## 连接路径 join

串联多个路径成为一个完整路径输出，例如将基础路径和文件夹路径、文件名称合成一个完整的文件路径：

```javascript
const {join} = require('path')
join('/foo', 'bar', 'baz/asdf', 'quux', '..'); // /foo/bar/baz/asdf
```

## 解析路径 resolve

可以把多个路径解析为一个绝对路径。它的功能就像对这些路径挨个不断进行“cd”操作，不同的地方在于不会去确认路径是否存在，如：

```javascript
const {resolve} = require("path")
resolve('foo/bar', './baz'); // /foo/bar/baz
resolve('foo/bar', '/tmp/file/'); // /tmp/file
```

# 全局对象

相比js本身的全局对象是window，nodejs的全局对象是global，而全局对象意味着不需要导入模块就只能直接调用。

## process

* 用于描述当前nodejs进程状态
* process.argv是命令行参数数组，第一个元素是node，第二个是脚本文件名，第三个开始是运行参数。

```javascript
  console.log(process.argv)
//  ['node','/byvoid/argv.js','2016','==v','Carbo Kuo']
```

### stdout & stdin 命令行输出输入流

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

## child_process

子进程模块是为了弥补js本身只支持单线程的问题，最常见的用途是用子进程调用命令行完成相关操作。

### 调用命令行

通过child_process.exec方法可在当前环境的命令行下执行命令。
例如查看文件一般使用`fs`模块，但我们也可以在命令行调用`dir`或`ls`命令完成，示例：

```javascript
const {exec} = require("child_process")
const childProcess = exec("dir", {
    cmd: "./log"  //  在上一级路径执行命令
}, (err, stdout, stderr) => {
    //  执行命令出错
    if (err) throw err
    //  执行命令后的获取到的命令行输出
    console.log(`stdout: ${stdout}`)
    //  执行命令后的错误输出
    console.error(`stderr: ${stderr}`)
})
//  每次命令行有输出就会触发
childProcess.stdout.on("data", chunk => {
    console.log(chunk)
})
//  命令行输出结束
childProcess.stdout.on("end", () => {
    console.log("输出结束")
})
```

要注意通过on监听输出流事件能获取到命令行持续输出的内容，适合耗时长或者保持运行不会结束的命令，如压缩、调试等。
而exec中的callback回调只在命令结束后才会调用，适合快速或只需要拿到结果的命令。
还有，查看文件一般使用`fs`模块进行处理，这里只是做个简单示范。

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