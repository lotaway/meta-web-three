@[TOC](WebAssembly学习-编写一个WebAssembly方法并放到网站上调用)

# 介绍

WebAssembly是一种可以将程序代码编译成二进制文件供js或Nodejs调用，以便在安全的前提下完成耗时的运算。
WebAssembly于2015年首次发布，第一次演示是在Firefox、Chrome和Edge上执行Unity的Angry
Bots游戏，是继HTML、CSS和JavaScript之后的第四种Web语言。到目前为止，94%的浏览器已经支持WebAssembly。
它能够在浏览器中实现接近本机的执行速度，使得我们有可能将桌面应用(如AutoCAD)甚至电子游戏(如《Doom 3》)移植到 Web。
应用领域包括游戏、音视频解编码、数据压缩、3D模型渲染、加密运算、AR/VR合成与识别等，对速度有要求的网页后端程序（为减少网络带来的延迟）也可移植到前台直接使用。
[第三方文档](https://www.wasm.com.cn)

# 为什么要学习

目前ServerLess和云服务、微服务的趋势越来越明显，后端程序被切割成了众多不同难度级别的服务。
有纯操作系统可放置完整应用程序的，类似自己搭建服务器。
有单单只有数据库或者Docker容器、或Nginx网站服务的。
也有更简化的只有云函数调用的。
目前外出就业，全栈开发甚至是前端开发岗位都要求对亚马逊AWS、阿里云或腾讯云有使用经验，对Window或Linux的操作要求变成了满足基础使用即可。
加上元宇宙的到来，3D模型、人工智能、区块链都将下放到前端完成，前端的比重越来越离谱，任务越来越复杂，如果有WebAssembly负责前端js不擅长的高效运算那将是强而有力的补足。
甚至一小部分企业已经开始招聘懂C/Rust+WebAssembly的前端/全栈开发工程师了，现在先行了解也无妨。

# 开发

虽然理论上可以使用任意语言如C/C++/Rust/Go编写wasm代码，但还必须编译工具支持，目前已知的工具：

* Web Assembly Edge
* [Emscripten](https://github.com/emscripten-core/emsdk)，可编译C/C++
* [Web Assembly Explorer](https://mbebenita.github.io/WasmExplorer/)，可编译C/C++
* [Wasm Fiddle](https://wasdk.github.io/WasmFiddle/)，可编译C
* [Rust Wasm-Pack](https://rustwasm.github.io/wasm-pack)
* [WebAssembly.studio](https://webassembly.studio)，可编译C/Rust/Wat
* [Wat2Wasm](https://webassembly.github.io/wabt/demo/wat2wasm/)，可编译Wat
* [Go](https://golang.google.cn/dl)，Go官方从1.1版本就开始支持编译成wasm
  接下来用C++语言实现一个斐波数列的计算：

```bash
int fib(int n)
{
   if (n <= 1)
      return n;
   return fib(n-1) + fib(n-2);
}
```

# 编译

使用程序完成编译，编译后将生成一个.wasm后缀的文件，例如`math.wasm`。

# 加载

## 用于Js

若是在前端使用，创建一个scripts.js文件，用于加载.wasm文件：

```javascript
let math;

// 写一个通用方法用于加载不同的WebAssembly模块
function loadWebAssembly(fileName) {
    return fetch(fileName)
        .then(response => response.arrayBuffer())
        .then(buffer => WebAssembly.compile(buffer)) // Buffer converted to Web Assembly 
        .then(module => {
            return new WebAssembly.Instance(module)
        }); // Instance of Web assmebly module is returened 
};

//We call the function for math.wasm for the given Instance. 
const Wasm = {}
loadWebAssembly('math.wasm')
    .then(instance => {
        //  这里加载进来的函数名称是根据编译步骤决定的`_Z3fibi`，而非开发步骤中你所定义的函数名`fib`
        Wasm.fibc = instance.exports._Z3fibi;

    });
```

## 用于Nodejs

Nodejs可以直接从本地加载文件，而不需要通过接口获取：

```javascript
const fs = require('fs')
const Wasm = {}
const buf = fs.readFileSync('./math.wasm')
const loadWebAssembly = WebAssembly.instantiate(new Uint8Array(buf)).then(res => {
    Wasm.fibc = res.instance.exports._Z3fibi
})
```

# 调用

```javascript
Wasm.fibc(45)
```

以上方法将花费9秒左右完成计算并输出结果，如果用纯js编写类似的函数，调用后将花费13秒左右方可输出结果。
