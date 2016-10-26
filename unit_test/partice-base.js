/**
    基础模块练习
 */

// 网络响应模块
var http = require('http');
/* 显性写法：
 var server = new http.Server();
 server.on('request',function(req,res){})
 * */
var server = http.createServer(function (req, res) {
    //req即http.ServerRequest
    //res即http.ServerResponse
    var qs = require('querystring');
    var post = '';
    req.on('data', function (chunk) {
        post += chunk;
    });
    req.on('end', function () {
        post = qs.parse(post);
    });
    /*
     ServerRequest事件：
     data：当请求体数据到来时，该事件被触发。该事件提供一个参数chunk，表示接收到的数据。该事件可能会被调用多次。如果该事件没有被监听，那么请求体将会被抛弃。
     end：当请求体数据传输完成时，该事件被触发，此后将不会再有数据到来。
     close： 用户当前请求结束时，该事件被触发。不同于 end，如果用户强制终止了传输，也还是调用close。
     ServerRequest属性：
     complete 客户端请求是否已经发送完成
     httpVersion HTTP协议版本，通常是1.0或1.1
     method HTTP 请求方法，如GET、POST、PUT、DELETE等url原始的请求路径，例如/static/image/x.jpg 或/user?name=byvoid
     headers HTTP 请求头
     trailers HTTP 请求尾（不常见）
     connection 当前HTTP连接套接字，为net.Socket的实例
     socket connection属性的别名
     client client 属性的别名*/
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('<span class="response">This is the response content.</span>');
    res.end('<p class="end">It\'s end.</p>');   //结束发送
});
server.listen(3000);
console.log("HTTP serrver is listening at port 3000");

//  安全的网络响应（提供证书）
var https = require('https');
var fs = require('fs');
var options = {
    key: fs.readFileSync('ssh_key.pem'),
    cert: fs.readFileSync('ssh_cert.pem')
};
https.createServer(options, function (req, res) { /*相同*/ });

//  在客户端发送POST请求
var http2 = require('http');
var qs2 = require('querystring');
var contents = qs2.stringify({
    //发送的内容
    name: 'lotaway',
    email: '576696294@qq.com',
    address: 'shantou,China'
});
var req = http2.request({
        //  请求的选项
        //host: 'www.lotaway.com',    //请求网站的域名或IP地址
        path: '/api/post/test?keyword=1', //请求的相对路径和数据参数
        method: 'POST', //请求方法，默认为GET
        headers: {  //请求头
            'Content-Type': 'application/json',
            'Content-Length': contents.length
        }
    },
    function (res) {    //  接收数据的回调函数，为http.ClientRequest的实例
        res.setEncoding('utf8');
        res.on('data', function (data) {
            console.log(data);
        });
    }
);
req.write(contents);
req.end();  //结束请求

// 若使用GET请求方法，有非常简便的写法：
var http3 = require('http');
http3.get({host:'www.lotaway.com'}, function (res) {
    res.setEncoding('utf8');
    res.on('data', function (data) {
        console.log(data);
    });
});




// fs模块是文件操作的封装，提供了文件读取、写入、更名、删除、遍历目录、链接等POSIX操作，并且都有同步和异步方法。以下是异步读取文件：
var fs = require('fs');
fs.readFile('forRead.txt', 'utf-8', function (err, data) {
    if (err) {
        //  没有错误时err为nll或undefined，有错误发生时为Error对象实例
        console.error(err);
    }
    else {
        console.log(data);
    }
});
console.log('finish set the I/O listener, going to read and trigger the event...');

//	文件读取：同步方式
var fs_sync = require('fs');
var data = fs_sync.readFileSync('forRead.txt', 'utf-8');
console.log(data + '\n----- read and show the file, then console this.');

/** POSIX open函数的封装
 * （一般来说，除非必要，否则不要使用这种方式读取文件，因为它要求你手动管理缓冲区和文件指针，尤其是在你不知道文件大小的时候，这将会是一件很麻烦的事情。）
 接受两个必选参数，path 为文件的路径，flags可以是以下值。
 r ：以读取模式打开文件。
 r+ ：以读写模式打开文件。
 w ：以写入模式打开文件，如果文件不存在则创建。
 w+ ：以读写模式打开文件，如果文件不存在则创建。
 a ：以追加模式打开文件，如果文件不存在则创建。
 a+ ：以读取追加模式打开文件，如果文件不存在则创建。
 mode 参数用于创建文件时给文件指定权限，默认是0666。
 回调函数将会传递一个文件描述符fd。
 1、文件权限指的是POSIX操作系统中对文件读取和访问权限的规范，通常用一个八进制数来表示。例如0754表示文件所有者的权限是7（读、写、执行），同组的用户权限是5（读、执行），其他用户的权限是4（读），写成字符表示就是 -rwxr-xr--。
 2、文件描述符是一个非负整数，表示操作系统内核为当前进程所维护的打开文件的记录表索引。

 fs.read(fd, buffer, offset, length, position, [callback(err, bytesRead,
 buffer)]) 是 POSIX read 函数的封装，相比 fs.readFile 提供了更底层的接口。 fs.read
 的功能是从指定的文件描述符 fd 中读取数据并写入 buffer 指向的缓冲区对象。 offset 是
 buffer 的写入偏移量。 length 是要从文件中读取的字节数。 position 是文件读取的起始
 位置，如果 position 的值为 null，则会从当前文件指针的位置读取。回调函数传递
 bytesRead 和 buffer，分别表示读取的字节数和缓冲区对象。
 */
fs.open('forRead.txt', 'r', function (err, fd) {
    if (err) {
        console.error(err);
        return;
    }
    var buf = new Buffer(8);
    fs.read(fd, buf, 0, 8, null, function (err, bytesRead, buffer) {
        if (err) {
            console.error(err);
            return;
        }
        console.log('bytesRead:' + bytesRead);
        console.log(buffer);
        /*运行结果则是：
         bytesRead: 8
         <Buffer 54 65 78 74 20 e6 96 87>*/
    });
});

/*
 fs所有函数的定义和功能表
 功能                 异步函数                                                     同步函数
 打开文件             fs.open(path,flags, [mode], [callback(err,fd)])            fs.openSync(path, flags, [mode])
 关闭文件             fs.close(fd, [callback(err)])                              fs.closeSync(fd)
 读取文件（文件描述符） fs.read(fd,buffer,offset,length,position,[callback(err, bytesRead, buffer)])    fs.readSync(fd, buffer, offset,length, position)
 写入文件（文件描述符） fs.write(fd,buffer,offset,length,position,[callback(err, bytesWritten, buffer)])    fs.writeSync(fd, buffer, offset,length, position)
 读取文件内容          fs.readFile(filename,[encoding],[callback(err, data)])     fs.readFileSync(filename,[encoding])
 写入文件内容          fs.writeFile(filename, data,[encoding],[callback(err)])    fs.writeFileSync(filename, data,
 [encoding])
 删除文件              fs.unlink(path, [callback(err)])                           fs.unlinkSync(path)
 创建目录              fs.mkdir(path, [mode], [callback(err)])                    fs.mkdirSync(path, [mode])
 删除目录              fs.rmdir(path, [callback(err)])                            fs.rmdirSync(path)
 读取目录              fs.readdir(path, [callback(err, files)])                   fs.readdirSync(path)
 获取真实路径           fs.realpath(path, [callback(err,resolvedPath)])            fs.realpathSync(path)
 更名                  fs.rename(path1, path2, [callback(err)])                   fs.renameSync(path1, path2)
 截断                  fs.truncate(fd, len, [callback(err)])                      fs.truncateSync(fd, len)
 更改所有权             fs.chown(path, uid, gid, [callback(err)])                  fs.chownSync(path, uid, gid)
 更改所有权（文件描述符） fs.fchown(fd, uid, gid, [callback(err)])                   fs.fchownSync(fd, uid, gid)
 更改所有权（不解析符号链接）fs.lchown(path, uid, gid, [callback(err)])              fs.lchownSync(path, uid, gid)
 更改权限               fs.chmod(path, mode, [callback(err)])                     fs.chmodSync(path, mode)
 更改权限（文件描述符）   fs.fchmod(fd, mode, [callback(err)])                      fs.fchmodSync(fd, mode)
 更改权限（不解析符号链接）fs.lchmod(path, mode, [callback(err)])                    fs.lchmodSync(path, mode)
 获取文件信息            fs.stat(path, [callback(err, stats)])                     fs.statSync(path)
 获取文件信息（文件描述符）fs.fstat(fd, [callback(err, stats)])                      fs.fstatSync(fd)
 获取文件信息（不解析符号链接）fs.lstat(path, [callback(err, stats)])                 fs.lstatSync(path)
 创建硬链接              fs.link(srcpath, dstpath, [callback(err)])                fs.linkSync(srcpath, dstpath)
 创建符号链接           fs.symlink(linkdata, path, [type],[callback(err)])         fs.symlinkSync(linkdata, path,[type])
 读取链接               fs.readlink(path, [callback(err,linkString)])              fs.readlinkSync(path)
 修改文件时间戳         fs.utimes(path, atime, mtime, [callback(err)])              fs.utimesSync(path, atime, mtime)
 修改文件时间戳（文件描述符）fs.futimes(fd, atime, mtime, [callback(err)])            fs.futimesSync(fd, atime, mtime)
 同步磁盘缓存              fs.fsync(fd, [callback(err)])                            fs.fsyncSync(fd)
 */




//	事件监听器
var ee = new require('events');
var event = new ee.EventEmitter();
event.on('custom-Event', function (arg1, arg2) {    //注册事件监听器，设置自定义事件和回调方法,可以设置多个事件，每个事件可以有多个回调方法
    console.log("trigger the event with arg1:" + arg1 + "and arg2:" + arg2);
});
event.once('only first', function () {
    console.log("only trigger on the first time");
});
event.on('error', function () {
    console.log("into the error event");
//    一般都要设置错误回调，否则错误时会退出程序并打印调用栈
});
setTimeout(function () {
    event.emit('custom-Event', 'A1', 'A2');    //触发事件
}, 2000);
//EventEmitter.removeListener(event, listener)移除指定事件的某个监听器，listener必须是该事件已经注册过的监听器。
event.removeAllListeners(); // 移除所有事件的所有监听器，参数可以指定event，则移除指定事件的所有监听器。


// 可以自定义模块文件：classroom.js , 然后使用require调用
var cr = require('./classroom');
//一般接口调用类方法
//cr.addTeacher('Mommy Bob');
//  用module.exports = obj 方法修改了接口对象时，要创建实例调用
var cr_obj = new cr();
cr_obj.addT('Larfer');


/**
 * 可以把许多模块文件集合成包，作为一个大组件来调用
 * 创建并将 模块文件 interface.js 放入 lib 子文件夹下。
 然后在somepackage 文件夹下，我们创建一个叫做 package.json 的文件写入interface.js所在路劲以同样的方式再次调用这个包，依然可以正常使用。
 Node.js 在调用某个包时，会首先检查包中 package.json 文件的 main 字段，将其作为包的接口模块，如果 package.json 或 main 字段不存在，会尝试寻找 index.js 或 index.node 作为包的接口。
 */
var pg = require('./somepackage/lib/interface');
pg.show();

/**
 * 核心模块由一些精简高效的库组成，为nodejs提供了基本的API，常用内容介绍：
 * 1、全局对象
 * 2、常用工具；
 * 3、事件机制；
 * 4、文件系统访问；
 * 5、HTTP服务器与客户端
 * */
//相比js本身的全局对象是window，nodejs的全局对象是global
//process：用于描述当前nodejs进程状态
//process.argv是命令行参数数组，第一个元素是node，第二个是脚本文件名，第三个开始是运行参数。
console.log(process.argv);
//['node','/byvoid/argv.js','2016','==v','Carbo Kuo']

//process.stdout是标准输出流，而process.stdin是标准输入流，但初始时是暂停的，要输入数据需要恢复流并配上回调函数：
process.stdin.resume();
process.stdin.on('data', function (data) {
    process.stdout.write('read from console:' + data.toString());
});

// nextTick()功能是为事件循环设置一项任务，为了把事件拆解成各小的部分，避免一个事件占用太多CPU时间，这比setTimeout(fn,0)的效率高得多
function doSomething(callback) {
    console.log("do something first");
    process.nextTick(callback);
}
function somethingElse() {
    console.log("do something else");
}
doSomething(somethingElse);

// 还有precess.platform,process.pid,process.execPath,process.memoryUsage()等方法，以及POSIX进程信号响应机制