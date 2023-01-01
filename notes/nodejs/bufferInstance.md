@[TOC](Buffer的学习)
//  写入字符串
var buf = new Buffer("123456789");
console.log(buf.length);

//  先确定长度,写入字符串时会丢弃多余字符
var buf2 = new Buffer(8);
//  写入字符串，参数是：源Buf（必须），偏移量，长度，编码（默认utf8）
buf2.write('123456789');
console.log(buf2.length);

//  更常用的是写入数组，可以直接通过下标调用
var buf3 = new Buffer([0, 1, 2, 3]);
console.log(buf3[0]);

//  复制，需要的参数是：源Buffer（必须），目标写入位置，源读取位置，源结束位置
var buf4 = new Buffer();
buf4.copy(buf3, 0, 0, buf3.length);

//  更多...
//buffer.toString(encoding,start=0,end=buffer.length);
//buffer.slice(start,end);
//buffer.compare(otherBuffer);
//buffer.equal(otherBuffer);
//buffer.fill(value,offset,end);

const fs = require('fs');
fs.readFile('./public/images/logo.png', function (err, buffer) {
    console.log(Buffer.isBuffer(buffer));

    fs.writeFile('logo-buffer.png', buffer, function (err) {
        if (err) console.log(err);
    });

    var b64 = buffer.toString('base64');
    console.log(b64);

    var di = new Buffer(b64, 'base64');
    console.log(Buffer.compare(buffer, di));

    fs.writeFile('log-decoded.png', di, function (err) {
        if (err) {
            console.log(err);
        }
    });

});

//     使用流输出，可断续
const fs = require('fs');
var reStm = fs.readFileSync('./public/images/logo.png');
reStm
    //  数据传输中
    .on('data', function (chunk) {
        console.log(Buffer.isBuffer(chunk));

        //  流暂停与继续
        /*reStm.pause();
         setTimeout(function () {
         reStm.resume();
         });*/

    })
    //  可读
    .on('readable', function () {
        console.log('data readable');
    })
    //  数据传输完成
    .on('end', function () {
        console.log('data end');
    })
    //  异常
    .on('error', function (err) {
        console.log(err);
    })
    //  流传输关闭
    .on('close', function () {
        console.log('steam close');
    });

//  输入输出流
var readStream = fs.createReadStream('./public/audio/1.mp4');
var writeStream = fs.createWriteStream('./public/write/1.mp4');

readStream
    .on('data', function (chunk) {
        //  上一次输入流未完成，仍有数据在缓存中
        if (writeStream.write(chunk) === false) {
            console.log('data still cached');
            readStream.pause();
        }
    })
    .on('edn', function () {
        //  读取结束时关闭输入流
        writeStream.end();
    });
//  输入流完成缓存写入后触发事件
writeStream.on('drain', function () {
    console.log('data drains');
    readStream.resume();
});