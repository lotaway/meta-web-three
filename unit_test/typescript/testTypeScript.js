/// <reference path="public.ts" />
// 需要依赖的文件，如果那些文件不存在则编译不通过（可尝试删除Validation.ts）
var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
var isDone = false, isNumber = 6, isfloat = 6.01, name = "bob", notSure = 4, //  可动态赋值为任意类型
list = [1, 2, 3], names = ["阿龙", "阿猫", "阿狗"], arr = [1, true, "free"], // 任意类型数组，有错误是因为需要指定类型为泛型<any>
arr2 = [1, true, "free"];
var Color;
(function (Color) {
    Color[Color["Red"] = 1] = "Red";
    Color[Color["Green"] = 2] = "Green";
    Color[Color["Blue"] = 3] = "Blue";
})(Color || (Color = {}));
var c = Color.Green;
var d = Color[2]; //  不同的取值方式
function func() {
    if (true) {
        var x = 0;
    }
    console.log("This is my warning message");
}
var Animal = (function () {
    function Animal(theName) {
        this.name = theName;
    }
    Animal.prototype.move = function (meters) {
        document.write(this.name + " moved " + meters + "m.");
    };
    Animal.prototype.fun = function () {
        //定义该方法所要实现的功能
    };
    Animal.prototype.say = function () {
        //定义该方法所要实现的功能
        return "返回值"; // 用return关键字返回函数值
    };
    Animal.prototype.fun2 = function () {
        //   类成员默认为public，定义为private时只能在内部使用
    };
    Animal.prototype.func3 = function () {
        //    定义为protected时在内部和继承类使用
    };
    Animal.count = 10;
    return Animal;
})();
var Snake = (function (_super) {
    __extends(Snake, _super);
    function Snake(name) {
        _super.call(this, name); //调用基本类的构造函数
    }
    Snake.prototype.move = function () {
        document.write("Slithering...<br>");
        _super.prototype.move.call(this, 5); //调用基类的方法
    };
    return Snake;
})(Animal);
var sam = new Snake("Python"); //声明Snake类
sam.move();
var Dragon = (function () {
    function Dragon() {
    }
    return Dragon;
})();
var Chicken = (function (_super) {
    __extends(Chicken, _super);
    function Chicken(name) {
        _super.call(this);
        this.name = name;
    }
    Chicken.prototype.fun4 = function () {
        console.log(this.name);
    };
    return Chicken;
})(Dragon);
function createSquare(config) {
    var newSquare = { color: "white", area: 100 };
    if (config.color) {
        newSquare.color = config.color;
    }
    if (config.width) {
        newSquare.area = config.width * config.width;
    }
    return newSquare;
}
var mySquare = createSquare({ color: "black" }); //定义含有接口中属性的对象
document.write(mySquare.color); //结果为： black
var mySearch;
mySearch = function (source, subString) {
    var result = source.search(subString); //调用系统方法search查找字符串的位置
    if (result == -1) {
        return false;
    }
    else {
        return true;
    }
};
var myArray;
myArray = ["Bob", "Fred"];
var A = (function () {
    function A() {
    }
    A.prototype.print = function () {
        document.write("实现接口");
    };
    return A;
})();
var B = new A();
B.print();
//  泛型即参数化类型，也就是说所操作的数据类型被指定为一个参数。这种参数类型可以用在类、接口和方法的创建中。
var BaseType = (function () {
    function BaseType() {
    }
    return BaseType;
})();
var ParamType = (function (_super) {
    __extends(ParamType, _super);
    function ParamType() {
        _super.apply(this, arguments);
        this.BizType = 'Po';
    }
    return ParamType;
})(BaseType);
var AO = (function () {
    function AO(OpCode, Context) {
        this.OpCode = OpCode;
        this.Context = Context;
    }
    return AO;
})();
var PO = (function (_super) {
    __extends(PO, _super);
    function PO(OpCode, Context) {
        _super.call(this, OpCode, Context);
    }
    PO.prototype.getContext = function () {
        return this.Context;
    };
    return PO;
})(AO);
//  命名空间，代码内部化
var frameWork;
(function (frameWork) {
    var Utils = (function () {
        function Utils() {
        }
        Utils.isNullOrEmtry = function (input) {
            if (input === null || input === "") {
                return true;
            }
            return false;
        };
        return Utils;
    })();
    console.log(Utils.isNullOrEmtry('test')); //  正确
})(frameWork || (frameWork = {}));
//console.log(frameWork.Utils.isNullOrEmtry('test')); //  错误，不能在外部调用命名空间内代码
//  复用模块，这样可以一个模块名能在全部文件中调用，并且可继承、可重复定义
// 使用 module 关键字来定义模块，并在末尾加花括号即可用； 用export 关键字使接口、类等成员对模块外可见。
var Validation;
(function (Validation) {
    var lettersRegexp = /^[A-Za-z]+$/;
    var numberRegexp = /^[0-9]+$/;
    var LettersOnlyValidator = (function () {
        function LettersOnlyValidator() {
        }
        LettersOnlyValidator.prototype.isAcceptable = function (s) {
            return lettersRegexp.test(s);
        };
        return LettersOnlyValidator;
    })();
    Validation.LettersOnlyValidator = LettersOnlyValidator;
    var ZipCodeValidator = (function () {
        function ZipCodeValidator() {
        }
        ZipCodeValidator.prototype.isAcceptable = function (s) {
            return s.length === 5 && numberRegexp.test(s);
        };
        return ZipCodeValidator;
    })();
    Validation.ZipCodeValidator = ZipCodeValidator;
})(Validation || (Validation = {}));
var strings = ['Hello', '98052', '101'];
var validators = {};
validators['ZIP code'] = new Validation.ZipCodeValidator(); //使用模块中的类
validators['Letters only'] = new Validation.LettersOnlyValidator();
// 显示匹配结果
for (var i = 0; i < strings.length; i++) {
    for (var pName in validators) {
        document.write('"' + strings[i] + '" ' + (validators[pName].isAcceptable(strings[i]) ? ' matches ' : ' does not match ') + pName + "<br>"); // 使用方法
    }
}
//# sourceMappingURL=testTypeScript.js.map