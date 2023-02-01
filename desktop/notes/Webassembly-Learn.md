@[TOC](WebAssembly学习-编写一个WebAssembly方法并放到网站上调用)

# 介绍

WebAssembly是一种将桌面或后端程序编写的代码放到网站上给前端脚本调用，在安全的前提下完成耗时的运算。
应用领域包括游戏、音视频解编码、数据压缩、3D模型、合成与识别等，对速度有要求的后端程序可移植到前台直接使用。
[第三方文档](https://www.wasm.com.cn)

# 为什么要学习

目前ServerLess的趋势越来越明显，后端程序被切割成了众多不同难度级别的服务。
有纯操作系统可放置完整应用程序的，类似自己搭建服务器。
有单单只有数据库或者Docker容器、或Nginx网站服务的。
也有更简化的只有云函数调用的。
目前外出就业，全栈开发甚至是前端开发岗位都要求对亚马逊AWS云服务、阿里云服务或腾讯云服务有使用经验，对Window或Linux的操作要求变成了满足基础使用即可。
加上元宇宙的到来，3D建模、人工智能、区块链都放到前端使用，前端的比重越来越离谱，任务越来越复杂，如果有WebAssembly负责前端javascript不擅长的高速计算是强而有力的补足。

# 开发

使用C++/Rust/Go开发工具编写代码，如：

```bash
// 使用C++语言实现一个斐波数列的计算
int fib(int n)
{
   if (n <= 1)
      return n;
   return fib(n-1) + fib(n-2);
}
```

# 编译

使用Web Assembly Explorer或者Web Assembly Edge等程序完成编译，编译后将生成一个.wasm后缀的文件，例如`math.wasm`。

# 加载

创建一个scripts.js文件，用于加载.wasm文件：

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

# 调用

```javascript
Wasm.fibc(45)
```

以上方法将花费9秒左右完成计算并输出结果，如果用纯js编写类似的函数，调用后将花费13秒左右方可输出结果。