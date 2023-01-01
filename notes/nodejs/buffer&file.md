@[TOC](Nodejs学习笔记 - Buffer和文件读写学习)

# Buffer介绍

`Buffer`既一种二进制的流，大部分数据传输时实际就是作为`Buffer`存在，包括但不限于HTTP上传下载文件、视音频播放、本地文件读取与写入等。
掌握`Buffer`和流的使用，充实基础才能解决实际面临的数据问题

# 字符串转换成Buffer

通过`new Buffer(string)`的方式可以将内容变成Buffer格式
参数：源Buf（必须），偏移量，长度，编码（默认utf8）

```javascript
var strBuffer = new Buffer("123456789")
console.log(strBuffer.length)   //  和数组一样通过length获取长度
```

# 指定Buffer长度

可以通过`new Buffer(number)`指定固定长度，当写入字符串内容时会丢弃多余字符

```javascript
var buf2 = new Buffer(8)
buf2.write('123456789')
console.log(buf2.length)
//  获取到的长度固定是8
```

# 数组转换成Buffer

更常用的是写入数组，可以直接通过下标调用

```javascript
var buf3 = new Buffer([0, 1, 2, 3])
console.log(buf3[0])
//  获取到的内容为原数组索引0位置的内容：0
```

# 复制Buffer

复制需要的参数是：源Buffer（必须），目标写入开始位置，源读取开始位置，源结束位置

```javascript
let originBuffer = new Buffer("这就是要复制的内容")
let targetBuffer = new Buffer()
targetBuffer.copy(originBuffer, 0, 0, originBuffer.length)
```

# 更多Buffer方法

```javascript
buffer.toString(encoding, start = 0, end = buffer.length)
buffer.slice(start, end)
buffer.compare(otherBuffer)
buffer.equal(otherBuffer)
buffer.fill(value, offset, end)
```

# 文件操作

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

## fs所有函数的定义和功能表

| 功能   | 异步函数 | 同步函数                                              |
|------| --- |---------------------------------------------------|
| 打开文件 | fs.open(path,flags, [mode], [callback(err,fd)]) | fs.openSync(path, flags, [mode])                  |
| 关闭文件 | fs.close(fd, [callback(err)]) | fs.closeSync(fd)                                  |
| 读取文件（文件描述符） | fs.read(fd,buffer,offset,length,position,[callback(err, bytesRead, buffer)]) | fs.readSync(fd, buffer, offset,length, position)  
| 写入文件（文件描述符） | fs.write(fd,buffer,offset,length,position,[callback(err, bytesWritten, buffer)]) | fs.writeSync(fd, buffer, offset,length, position) |
| 读取文件内容 | fs.readFile(filename,[encoding],[callback(err, data)]) | fs.readFileSync(filename,[encoding])              |
| 写入文件内容 | fs.writeFile(filename, data,[encoding],[callback(err)]) | fs.writeFileSync(filename, data,[encoding]) |
| 删除文件 | fs.unlink(path, [callback(err)]) | fs.unlinkSync(path)                               |
| 创建目录 | fs.mkdir(path, [mode], [callback(err)]) | fs.mkdirSync(path, [mode])                        |
| 删除目录 | fs.rmdir(path, [callback(err)]) | fs.rmdirSync(path)                                |
| 读取目录 | fs.readdir(path, [callback(err, files)]) | fs.readdirSync(path)                              |
| 获取真实路径 | fs.realpath(path, [callback(err,resolvedPath)]) | fs.realpathSync(path)                             |
| 更名 | fs.rename(path1, path2, [callback(err)]) | fs.renameSync(path1, path2)                       |
| 截断 | fs.truncate(fd, len, [callback(err)]) | fs.truncateSync(fd, len)                          |
| 更改所有权 | fs.chown(path, uid, gid, [callback(err)]) | fs.chownSync(path, uid, gid)                      |
| 更改所有权（文件描述符） | fs.fchown(fd, uid, gid, [callback(err)]) | fs.fchownSync(fd, uid, gid)                       |
| 更改所有权（不解析符号链接） | fs.lchown(path, uid, gid, [callback(err)]) | fs.lchownSync(path, uid, gid)                     |
| 更改权限 | fs.chmod(path, mode, [callback(err)]) | fs.chmodSync(path, mode)                          |
| 更改权限（文件描述符） | fs.fchmod(fd, mode, [callback(err)]) | fs.fchmodSync(fd, mode)                           |
| 更改权限（不解析符号链接） | fs.lchmod(path, mode, [callback(err)]) | fs.lchmodSync(path, mode)                         |
| 获取文件信息 | fs.stat(path, [callback(err, stats)]) | fs.statSync(path)                                 |
| 获取文件信息（文件描述符） | fs.fstat(fd, [callback(err, stats)]) | fs.fstatSync(fd)                                  |
| 获取文件信息（不解析符号链接） | fs.lstat(path, [callback(err, stats)]) | fs.lstatSync(path)                                |
| 创建硬链接 | fs.link(srcpath, dstpath, [callback(err)]) | fs.linkSync(srcpath, dstpath)                     |
| 创建符号链接 | fs.symlink(linkdata, path, [type],[callback(err)]) | fs.symlinkSync(linkdata, path,[type])             |
| 读取链接 | fs.readlink(path, [callback(err,linkString)]) | fs.readlinkSync(path)                             |
| 修改文件时间戳 | fs.utimes(path, atime, mtime, [callback(err)]) | fs.utimesSync(path, atime, mtime)                 |
| 修改文件时间戳（文件描述符） | fs.futimes(fd, atime, mtime, [callback(err)]) | fs.futimesSync(fd, atime, mtime)                  |
| 同步磁盘缓存 | fs.fsync(fd, [callback(err)]) | fs.fsyncSync(fd)                                  | 