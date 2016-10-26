/**  Created by lotaway on 2016/3/13.

 block scoping － 使用 let 、const 作用域，块辖域
 class、extends - 类、继承
 generators - 生成器
 collections － 集合、映射、弱集合、弱映射
 arrow functions － 箭头函数，简化匿名函数，指定词法作用域
 template strings － 模板字串
 promises － 用标准化了的方法进行延迟和异步计算
 symbols － 唯一的、不可修改的数据

 */

// 函数级别严格模式语法
'use strict';

// 关于ES6

//  导入其他文件中写好的内容
import * as inFunc from './someFunc.js'; //  导入全部内容，通过inFunc对象调用
console.log(inFunc());
console.log(inFunc.square(10)); // 100

import { default, square, diag } from './someFunc.js'; //  只导入默认内容、square和diag
console.log(square(10)); // 100
console.log(diag(4, 3));

+function ES6() {

    //var的作用域是函数，而let的作用域是块（if、for)
    var a = 5,
        b = 10;
    const c = 20;
    if (a === 5) {
        let a = 4; // 只作用于（if-block）块级内,若放在if-block之外，则对if-block内也不会造成影响，总结：let作用域不会跨越大括号{}
        var b = 1; // 作用于整个（function）函数内
        const c = 10;  //   错误，常量不能改变值
        console.log(a);  // 4
        console.log(b);  // 1
    }
    console.log(a); // 5
    console.log(b); // 1
    console.log(c); // 20


    // array.map
    // SyntaxEDIT：arr.map(callback[, thisArg])
    var numbers = [1, 4, 9];
    var roots = numbers.map(Math.sqrt);
    // or
    var roots = numbers.map(function (num) {
        return Math.sqrt(num);
    });
    // roots is now [1, 2, 3], numbers is still [1, 4, 9]
    var kvArray = [{key: 1, value: 10}, {key: 2, value: 20}, {key: 3, value: 30}];
    var reformattedArray = kvArray.map(function (obj) {
        var rObj = {};
        rObj[obj.key] = obj.value;
        return rObj;
    });
// reformattedArray is now [{1:10}, {2:20}, {3:30}], 
// kvArray is still [{key:1, value:10}, {key:2, value:20}, {key:3, value: 30}]


//    class 类声明，类内部默认为严格模式，类名注意大写
    class OddOne {
        // constructor 只能有一个构造方法，默认为return this，生成实例时自动调用
        oldFunc() {
            console.log('old function');
        }
    }
    class NewOne extends OddOne {
        // 类可以被继承，内部的方法可以重写
        constructor(potX, potY) {
            super();    //  执行父类的构造
            this.x = potX;
            this.y = potY;
            //return this; 默认已执行该语句，可以改为其他，这时新建的实例就不是本类的实例
        }

        //实例方法
        sayIt() {
            console.log("X point is " + this.x + " and Y point is " + this.y);
        }

        //静态方法，无需实例化就可调用，定义一些常用的数学函数等等
        static distance(a, b) {
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            return Math.sqrt(dx * dx + dy * dy);
        }
    }
    //类声明也可以用表达式写法：
    //var NewOne = class NewOne(){ ... }
    //调用类
    var p1 = new NewOne(1, 2);
    p1.sayIt();
    var p2 = new NewOne(3, 4);
    p2.oldFunc();
    NewOne.distance(p1, p2);


    //Map对象是一个简单的键值映射，可以放入字符串/对象/数组等作为键或值，利用set进行键值设置，利用get和键获取值
    var m = new Map([["name1", "mj"], ["name2", "cj"]]);
    m.get("name2"); // cj
    var d = {p: "Hello World"};
    m.set(d, "text content");
    m.get(d);   // "text content"
    console.log(m.size);     //  返回成员数量
    m.delete(d);    //删除键，返回布尔值表示删除是否成功
    m.has(d);    // 确认是否存在该键 返回布尔值
    m.clear();  // 清楚所有成员，无返回值
    //遍历方法
    m.keys();   //  返回键名的遍历器
    for (let key of m.keys()) {
        console.log(key);
    }
    m.values(); //  返回值的遍历器
    m.entries();    //  返回所有成员的的遍历器
    m.forEach(func(value, key), m); //  使用回调函数遍历
    // 也可以直接使用语句如：
    // m.forEach((value,key) => console.log(value * 2))

    /**
     *  WeakMap结构与Map结构基本类似，唯一的区别是它只接受对象作为键名（null除外），不接受其他类型的值作为键名，而且键名所指向的对象若在其他地方没有引用时就会自动回收释放资源。
     WeakMap没有size属性，只有四个方法可用：get()、set()、has()、delete()
     */
    var myMap = new WeakMap();
    var keyString = "a string";
    myMap.set(keyString, "和键'a string'关联的值");
    myMap.get(keyString);

    /**数据结构Set。它类似于数组，但是成员的值都是唯一的，没有重复的值。
     向Set加入值的时候，不会发生类型转换，所以5和"5"是两个不同的值。Set内部判断两个值是否不同，使用的算法类似于精确相等运算符（===），这意味着，两个对象总是不相等的。唯一的例外是NaN等于自身（精确相等运算符认为NaN不等于自身）。*/
    var s = new Set();
    [2, 3, 5, 4, 5, 2, 2].map(x => s.add(x));
    for (i of s) {
        console.log(i);  // 2 3 5 4
    }
    s.add("value"); //添加新的值
    if (s.has("value"))  //判断是否存在值
        s.delete("value");  //删除值
    console.log(s.size); //总数量
    s.clear();  //清除所有成员
    //set同样有keys、values、entires、forEach四种遍历方法

    var ws = new WeakSet(); // 类似于WeakMap相对于Map，WeakSet是相对于Set的。


    /**
     * Generator遍历器是一个状态机，封装了多个内部状态。
     * 相对于普通函数,遍历器以function后面的星号为区别,执行此函数会返回一个指针对象，用于遍历状态。
     * 以yield为暂停标记，每次执行指针对象的next()方法，就会从上一次所在位置继续执行到下一个yield或return为止。
     * next()返回一个对象，该对象的value属性就是当前yield语句的值，done属性为false则表示遍历未结束。
     * */
    /*function* helloWorldGenerator(i) {
        //这个函数有3个状态，2个yield和1个return
        yield 'hello';
        yield i + 1;
        return 'ending';
    }

    var hw = helloWorldGenerator(1); //执行只是返回指向内部状态的指针对象——遍历器对象（Iterator Object）
    hw.next();  //执行两个yield之间的语句，返回的对象value属性为"hello"，done属性为false

    //yield后面可以跟一个遍历器对象，同样加星号区别
    function* secGrator(i) {
        yield * helloWorldGenerator(i);
    }

    var gen = secGrator(2);*/


    /**
     * Promise对象，用来传递异步操作的消息，不受外界影响与不可被改变。
     * 有三个状态：Pending进行中、Resolved/Fulfilled已完成、Rejected失败。只会从Pending变成Resolved或Rejected。
     * 相对于Event事件，不会错过结果。
     * */
    var promise = new Promise(function (resolve, reject) {
        //    resolve和reject分别是完成和拒绝的回调函数，本身已定义好用于变换Promise的状态，也可以自定义，reject是非必须的
        //    定义异步操作
        if (true/* 操作成功*/) resolve(value);
        else reject(reason);
    });

    //实例生成后可以用then来定义完成和拒绝两个回调函数
    promise.then(function (value) {
        //    成功后的操作
    }, function (reason) {
        //    失败后的操作
    });

    promise.catch(function () {
        //    处理拒绝的情况，相当于 promise.then(undefined, function (reason) {});
    });

    //多个承诺组合，承诺p1,p2,p3都完成才算最终完成，此时所有返回值组成数组；若有拒绝的则只有第一个被拒绝的对象的返回值
    var pList = Promise.all([p1, p2, p3]);

    //多个承诺组合，但是只有任意一个返回完成/拒绝则传递完成。
    var forOne = Promise.race([p1, p2, p3]);

    //用来把对象或数组转成承诺（一般用于承诺组成），后面可以加then方法测试完成状态
    var target = 3;
    var inToP = Promise.resolve(target);

    Promise.reject("static rejected").then(function (reason) {
        console.log(reason);    //测试拒绝状态
    });


    /**
     * Symbol原始数据类型，表示独一无二的值。它是JavaScript语言的第七种数据类型，
     * 前六种是：Undefined、Null、布尔值（Boolean）、字符串（String）、数值（Number）、对象（Object）。
     * Symbol在创建时接受一个参数作为描述，就算描述相同也是不同的值。
     * */
    var s1 = Symbol("foo");
    var s2 = Symbol("foo");
    console.log(s1 === s2); // false
    //检查是否存在Symbol值，若不存在则创建一个，最后无论如何都返回该Symbol
    var s3 = Symbol.for("foo");
    var s4 = Symbol.for("foo");
    console.log(s3 === s4); // true
    //由于每个值都不相等，可以作为属性名，防止改写或覆盖，但不能用“.”方法设置或调用
    var a = {};
    a[s1] = 'Hello!';    // 第一种写法
    var a = {
        [s1]: 'Hello!'     // 第二种写法
    };
    Object.defineProperty(a, s1, {value: 'Hello!'});// 第三种写法
    console.log(a[mySymbol]); // 以上写法都得到同样结果 "Hello!"
    /**
     * Symbol作为属性名，该属性不会出现在for in、for of循环中，也不会被Object.keys()、Object.getOwnPropertyNames()返回。
     * 但是，它也不是私有属性
     */
        //返回一个当前对象的所有用作属性名的Symbol值的成员数组
    Object.getOwnPropertySymbols(a);


    /**
     =>箭头函数，主要功能是简化函数写法，如回调函数/监听函数
     箭头函数有几个使用注意点:
     （1）函数体内的this对象，绑定定义时所在的对象，而不是使用时所在的对象。
     （2）不可以当作构造函数，也就是说，不可以使用new命令，否则会抛出一个错误。
     （3）不可以使用arguments对象，该对象在函数体内不存在。如果要用，可以用Rest参数代替。
     （4）不可以使用yield命令，因此箭头函数不能用作Generator函数。
     （5）this对象的指向在箭头函数中是固定的。
     */
    var v1 = arg =>"result";
    // 以上语句相当于：
    var v1 = function (arg) {
        return "result";
    };
    //当没有参数或者有多个参数时，用小括号包起来：
    var v2 = () =>"result";
    var v3 = (a1, a2) =>"result";
    //当函数中没有语句或者有多个语句时用花括号包起来,注意对象本身会有一层花括号：
    var v4 = (a3, a4) => {
            console.log(a3 + a4), {"a3": a3, "a4": a4}
        }
        ;
    //特性（1）体现：由于用了箭头函数，this指向了定义时的实例对象，所以setInterval可以调用this属性
    function Timer() {
        this.seconds = 0
        setInterval(() => this.seconds++, 1000)
    }

    var timer = new Timer();
    setTimeout(() => console.log(timer.seconds), 3100);  // 3


    /**
     ``模板字符串，用反括号可以定义多行/跨行字符串，即可以保持换行，更可以加入表达式
     */
    var a = 5;
    var b = 10;
    var str1 = `string text ${ a + b }
    mutilate lines
    string text`;

    /**
     模板字符串的一种更高级的形式称为带标签的模板字符串。它允许您通过一个标签函数修改模板字符串的输出。
     标签函数的第一个参数是一个包含了字符串字面值的数组（在本例中分别为“Hello”和“world”）；
     第二个参数，在第一个参数后的每一个参数，都是已经被处理好的替换表达式（在这里分别为“15”和“50”）。最后函数可以返回处理好的字符串。
     标签函数的名称可以为任意的合法标示符。
     */
    function tag(strings) {
        console.log(arguments);    // { '0': [ 'Hello ', ' world ', '' ], '1': 15, '2': 50 }
        console.log(strings[0]);   // "Hello "
        console.log(strings[1]);   // " world "
        console.log(arguments[1]);  // 15
        console.log(arguments[2]);  // 50
        console.log(String.raw(strings));      //输出普通的字符串： "Hello world"
        return "Hubwiz!";
    }

    console.log(tag`Hello ${ a + b } world ${ a * b}`);  // "Hubwiz!"

    //\uxxxx形式表示一个字符，但只限于\u0000——\uFFFF之间的字符。超出这个范围的字符，必须用两个双字节的形式表达：
    var gu1 = "\uD842\uDFB7";  // "古"
    //新的表示法可以用大括号扩起码点，用单字节形式表达：
    //var gu2 = "\u{D842DFB7}";   // "古"
    //codePointAt可以传入字符位置参数，对字符串返回字符的码点。
    //gu2.codePointAt(0);

    //ES5提供String.fromCharCode方法，用于从码点返回对应字符，但是这个方法不能识别辅助平面的字符（编号大于0xFFFF）。
    String.fromCharCode(0x20BB7);   // "ஷ"
    //上面代码中，String.fromCharCode不能识别大于0xFFFF的码点，所以0x20BB7就发生了溢出，最高位2被舍弃了，最后返回码点U+0BB7对应的字符，而不是码点U+20BB7对应的字符。
    //ES6提供了String.fromCodePoint方法，可以识别0xFFFF的字符，弥补了String.fromCharCode方法的不足。在作用上，正好与codePointAt方法相反。
    String.fromCodePoint(0x20BB7);  // "𠮷"
    //注意，fromCodePoint方法定义在String对象上，而codePointAt方法定义在字符串的实例对象上。


    //除了indexOf方法，可以用来确定一个字符串是否包含在另一个字符串中，ES6又提供了三种新方法：
    var s = 'Hello world!';
    s.startsWith('Hello'); // 返回布尔值，表示参数字符串是否在源字符串的头部
    s.endsWith('!'); // 返回布尔值，表示参数字符串是否在源字符串的尾部
    s.includes('o'); // 返回布尔值，表示是否找到了参数字符串
    //这三个方法都支持第二个参数，表示开始搜索的位置。
    s.includes('o', 9);

//    Reflect提供了若干个能对任意对象进行某种特定的可拦截操作（interceptable operation）的方法。主要用来配合Proxy使用？14 个静态方法，它们的名字刚好和那 14 个代理处理器方法的名字相同，这 14 个方法中有几个刚好在 Object 对象身上也存在同名方法，虽然它们功能类似，但也存在细微差异。
    Reflect.apply(target,thisArgument,argumentsList);
    //对一个函数进行调用操作，同时可以传入一个数组作为调用参数。和 Function.prototype.apply() 功能类似。
    Reflect.construct(target,argumentsList);
    //对构造函数进行 new 操作，相当于执行 new target(...args)。
    Reflect.defineProperty(target,propertyKey,attributes);
    //和 Object.defineProperty() 类似。
    Reflect.deleteProperty(target,propertyKey);
    //删除对象的某个属性，相当于执行 delete target[name]。
    Reflect.enumerate(target);
    //该方法会返回一个包含有目标对象身上所有可枚举的自身字符串属性以及继承字符串属性的迭代器，for...in 操作遍历到的正是这些属性。
    Reflect.get(target,propertyKey,receiver);
    //获取对象身上某个属性的值，类似于 target[name]。
    Reflect.getOwnPropertyDescriptor(target,propertyKey);
    //类似于 Object.getOwnPropertyDescriptor()。
    Reflect.getPrototypeOf(target);
    //类似于 Object.getPrototypeOf()。
    Reflect.has(target,propertyKey);
    //判断一个对象是否存在某个属性，和 in 运算符 的功能完全相同。
    Reflect.isExtensible(target);
    //类似于 Object.isExtensible().
    Reflect.ownKeys(target);
    //返回一个包含所有自身属性（不包含继承属性）的数组。
    Reflect.preventExtensions(target);
    //类似于 Object.preventExtensions()。
    Reflect.set(target,propertyKey,value,receiver);
    //设置对象身上某个属性的值，类似于 target[name] = val。
    Reflect.setPrototypeOf(target,proto);
    //类似于 Object.setPrototypeOf()。


    /* Proxy(target,handler) 代理其功能非常类似于设计模式中的代理模式，该模式常用于三个方面：
    拦截和监视外部对对象的访问
    降低函数或类的复杂度
    在复杂操作前对操作进行校验或对所需资源进行管理*/
    function createValidator(target, validator) {
        //  公用的添加代理验证
        const argTypes = {
            pickyMethodOne: ["object", "string", "number"],
            pickyMethodTwo: ["number", "object"]
        };
        return new Proxy(target, {
            _validator: validator,
            set(target, key, value, proxy) {
                //  设置属性时会被这个set方法拦截
                if (target.hasOwnProperty(key)) {
                    let validator = this._validator[key];
                    if (!!validator(value)) {
                        //  验证通过时，调用Reflect进行正式的属性设置，@TODO 传入proxy是什么
                        return Reflect.set(target, key, value, proxy);
                    } else {
                        throw Error(`Cannot set ${key} to ${value}. Invalid.`);
                    }
                } else {
                    throw Error(`${key} is not a valid property`)
                }
            },
            get: function(target, key, proxy) {
                //  获取属性或者调用方法时会被这个方法拦截
                var value = target[key];
                return function(...args) {
                    var checkArgs = argChecker(key, args, argTypes[key]);
                    return Reflect.apply(value, target, args);
                };
            },
            has: function (target,key) {
            //    当使用例如："key" in proxyObject会调用此方法
            }
        //   @TODO 还有其他方法...
        });
    }
    function argChecker(name, args, checkers) {
        //  对每个属性进行值和类型检查，在get方法里调用
        for (var idx = 0; idx < args.length; idx++) {
            var arg = args[idx];
            var type = checkers[idx];
            if (!arg || typeof arg !== type) {
                console.warn(`You are incorrectly implementing the signature of ${name}. Check param ${idx + 1}`);
            }
        }
    }
    const personValidators = {
        //  自定义的验证器
        name(val) {
            //  每个属性名都有对应的验证方法，设置属性值时代理会调用
            return typeof val === 'string';
        },
        age(val) {
            return typeof age === 'number' && age > 18;
        }
    };
    class Person {
        //  定义一个类，使用代理并传入验证器
        pickyMethodOne(obj, str, num) { }
        constructor(validator,name, age) {
            this.name = name;
            this.age = age;
            return createValidator(this, validator);
        }
    }
    const bill = new Person(personValidators,'Bill', 25);
//  以下操作都会报错，因为没通过代理验证
    bill.name = 0;
    bill.age = 'Bill';
    bill.age = 15;
    bill.pickyMethodOne();  // You are incorrectly implementing the signature of pickyMethodOne. Check param 1  param 2  param 3
    bill.pickyMethodOne({}, "a little string", 123);    // No warnings logged

    //Proxy.revocable(target, handler)。这个函数一样创建代理，但是创建好的代理后续可被解除。（Proxy.revocable方法返回一个对象，该对象有一个.proxy属性和一个.revoke方法。）一旦代理被解除，它即刻停止运行并抛出所有内部方法。

    const {sensitiveData, revokeAccess} = Proxy.revocable({ username: 'devbryce' }, handler);
    //返回了无名对象，令常量sensitiveData是proxy属性，而revokeAccess是revoke属性
    console.log(sensitiveData.username); // logs 'devbryce'
    revokeAccess();
    console.log(sensitiveData.username); // TypeError: Revoked

    /*Decorator
    ES7 中实现的 Decorator，相当于设计模式中的装饰器模式。如果简单地区分 Proxy 和 Decorator 的使用场景，可以概括为：Proxy 的核心作用是控制外界对被代理者内部的访问，Decorator 的核心作用是增强被装饰者的功能。只要在它们核心的使用场景上做好区别，那么像是访问日志这样的功能，虽然本文使用了 Proxy 实现，但也可以使用 Decorator 实现，开发者可以根据项目的需求、团队的规范、自己的偏好自由选择。*/

    //@TODO Object.assign 原生的实现 mixin 对象的方法


    //解构赋值？ 解构赋值允许你使用类似数组或对象字面量的语法将数组和对象的属性赋给各种变量。这种赋值语法极度简洁，同时还比传统的属性访问方法更为清晰。

    //通常来说，你很可能这样访问数组中的前三个元素：
    var first = someArray[0];
    var second = someArray[1];
    var third = someArray[2];

    //如果使用解构赋值的特性，将会使等效的代码变得更加简洁并且可读性更高：
    var [first, second, third] = someArray;

    //对任意深度的嵌套数组进行解构：
    var [foo, [[bar], baz]] = [1, [[2], 3]];
    console.log(foo + bar + baz);   // 123

    //此外，你可以在对应位留空来跳过被解构数组中的某些元素：
    var [,,third] = ["foo", "bar", "baz"];
    console.log(third); // "baz"

    //而且你还可以通过“不定参数”模式捕获数组中的所有尾随元素：
    var [head, ...tail] = [1, 2, 3, 4];
    console.log(tail);  // [2, 3, 4]

}();