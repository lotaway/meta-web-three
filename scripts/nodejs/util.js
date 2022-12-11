/**
 * Created by lw on 2016/6/2.
 */
var util = require('util');

/*
util.inherits(constructor, superConstructor)是一个实现对象间原型继承 的函数。
*/
function Base() {
    this.name = 'base';
    this.base = 1991;
    this.sayHello = function () {
        console.log('Hello ' + this.name);
    };
}

Base.prototype.showName = function () {
    console.log(this.name);
};

function Sub() {
    this.name = 'sub';
}

util.inherits(Sub, Base);   //  让Sub继承Base
var baseObj = new Base();
console.log(baseObj);   //{ name: 'base', base: 1991, sayHello: [Function] }
var objSub = new Sub();
objSub.showName();  //  继承了原型方法
//objSub.sayHello();    //  出错
console.log(objSub);    //  { name: 'sub' } 不继承属性和实例方法的
/*我们定义了一个基础对象Base 和一个继承自Base 的Sub，Base 有三个在构造函数 内定义的属性和一个原型中定义的函数，通过util.inherits 实现继承。运行结果如下：
base
Hello base
{ name: 'base', base: 1991, sayHello: [Function] }
sub
{ name: 'sub' }
注意：Sub 仅仅继承了Base 在原型中定义的函数，而构造函数内部创造的 base 属 性和 sayHello 函数都没有被 Sub 继承。
同时，在原型中定义的属性不会被console.log 作 为对象的属性输出。如果我们去掉 objSub.sayHello(); 这行的注释，将会看到：
node.js:201
throw e; // process.nextTick error, or 'error' event on first tick
^
TypeError: Object #&lt;Sub&gt; has no method 'sayHello'
at Object.&lt;anonymous&gt; (/home/byvoid/utilinherits.js:29:8)
at Module._compile (module.js:441:26)
at Object..js (module.js:459:10)
at Module.load (module.js:348:31)
at Function._load (module.js:308:12)
at Array.0 (module.js:479:10)
at EventEmitter._tickCallback (node.js:192:40)*/

/*
 util.inspect(object,[showHidden],[depth],[colors])是一个将任意对象转换为字符串的方法，通常用于调试和错误输出。它至少接受一个参数 object，即要转换的对象。
 showHidden 是一个可选参数，如果值为 true，将会输出更多隐藏信息。
 depth 表示最大递归的层数，如果对象很复杂，你可以指定层数以控制输出信息的多少。如果不指定depth，默认会递归2层，指定为null表示将不限递归层数完整遍历对象。
 如果color 值为 true，输出格式将会以 ANSI 颜色编码*/
function Person() {
    this.name = 'byvoid';
    this.toString = function () {
        return this.name;
    };
}

var obj = new Person();
console.log(util.inspect(obj));
console.log(util.inspect(obj, true));
/*运行结果是：
{ name: 'byvoid', toString: [Function] }
{ toString:
{ [Function]
    [prototype]: { [constructor]: [Circular] },
    [caller]: null,
    [length]: 0,
    [name]: '',
    [arguments]: null },
    name: 'byvoid' }*/
util.isArray(object);
//如果给定的参数 "object" 是一个数组返回true，否则返回false。

util.isArray([]);
// true
util.isArray(new Array);
// true
util.isArray({});
// false
util.isRegExp(object);
//如果给定的参数 "object" 是一个正则表达式返回true，否则返回false。

util.isRegExp(/some regexp/);
// true
util.isRegExp(new RegExp('another regexp'));
// true
util.isRegExp({});
// false
util.isDate(object);
//如果给定的参数 "object" 是一个日期返回true，否则返回false。

util.isDate(new Date());
// true
util.isDate(Date());
// false (without 'new' returns a String)
util.isDate({});
// false
util.isError(object);
//如果给定的参数 "object" 是一个错误对象返回true，否则返回false。

util.isError(new Error());
// true
util.isError(new TypeError());
// true
util.isError({name: 'Error', message: 'an error occurred'});

// false