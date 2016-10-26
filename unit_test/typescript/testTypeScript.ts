/// <reference path="public.ts" />
// 需要依赖的文件，如果那些文件不存在则编译不通过（可尝试删除public.ts）

var isDone: boolean = false,
    isNumber: number = 6,
    isfloat: number = 6.01,
    name: string = "bob",
    notSure: any = 4,   //  可动态赋值为任意类型
    list: number[] = [1, 2, 3],
    names: string[] = ["阿龙", "阿猫", "阿狗"],
    arr: Array = [1, true, "free"], // 任意类型数组，有错误是因为需要指定类型为泛型<any>
    arr2: any[] = [1, true, "free"];
enum Color {
    Red = 1,　　　//枚举元素列表，若值留空，根据第一项或者从0开始递增
    Green,
    Blue
}
var c: Color = Color.Green;
var d: string = Color[2];   //  不同的取值方式


function func(): void {
    if (true) {
        let x: number = 0;
    }
    console.log("This is my warning message");
}

class Animal {

    static count = 10;

    name: string;  //定义类的属性

    constructor(theName: string) {
        this.name = theName;
    }

    move(meters: number) {
        document.write(this.name + " moved " + meters + "m.");
    }

    fun() { //定义了一个无返回值的方法
        //定义该方法所要实现的功能
    }

    public say(): string { //定义返回值类型为string的方法
        //定义该方法所要实现的功能
        return "返回值"; // 用return关键字返回函数值
    }

    private fun2() {
        //   类成员默认为public，定义为private时只能在内部使用
    }

    protected func3() {
        //    定义为protected时在内部和继承类使用
    }

}

class Snake extends Animal { //继承基类
    constructor(name: string) {
        super(name); //调用基本类的构造函数
    }

    move() { //重写了基类的方法
        document.write("Slithering...<br>");
        super.move(5); //调用基类的方法
    }
}
var sam = new Snake("Python"); //声明Snake类
sam.move();

abstract class Dragon { //  抽象类，由继承类实现
    abstract fun4();    //  只能定义在抽象类的抽象成员，由继承类实现
}
class Chicken extends Dragon {
    name: string;

    fun4() {    //  实现继承的抽象类的方法，需要定义在构函数之前
        console.log(this.name);
    }

    constructor(name: string) {
        super();
        this.name = name;
    }
}


//  接口，用于约束参数和类、函数需要实现的方法
interface SquareConfig { //定义了两个可选属性
    color?: string;
    width?: number;
}
function createSquare(config: SquareConfig): {color: string; area: number} { //定义函数printLabel,其参数类型为接口类型
    var newSquare = {color: "white", area: 100};
    if (config.color) {
        newSquare.color = config.color;
    }
    if (config.width) {
        newSquare.area = config.width * config.width;
    }
    return newSquare;
}

var mySquare = createSquare({color: "black"}); //定义含有接口中属性的对象
document.write(mySquare.color);   //结果为： black

//向对象思想中，接口的实现是靠类来完成的，而 function 作为一种类型，是不是能够实现接口呢?答案是肯定的。在 TypeScript 中，我们可以使用接口来约束方法的签名。

interface SearchFunc {
    (source: string, subString: string): boolean; //定义一个匿名方法
}

var mySearch: SearchFunc;
mySearch = function (source: string, subString: string) {  //实现接口
    var result = source.search(subString);  //调用系统方法search查找字符串的位置
    if (result == -1) {
        return false;
    }
    else {
        return true;
    }
};

//在前面一节中我们学习了接口定义方法类型，这一节我们来学习接口定义数组类型。在数组类型中有一个“index”类型其描述数组下标的类型，以及返回值类型描述每项的类型。如下：

interface StringArray { //定义数组接口
    [index: number]: string;  //每个数组元素的类型
}

var myArray: StringArray;
myArray = ["Bob", "Fred"];

//在C#和java中interface是很常使用的类型系统，其用来强制其实现类符合其契约。在TypeScript中同样也可以实现，通过类实现接口要用implements关键字。如下代码：

interface IPrint {
    print();
}

class A implements IPrint { //实现接口
    print() {  //实现接口中的方法
        document.write("实现接口");
    }
}

var B = new A();
B.print();


// 和类一样，接口也能继承其他的接口。这相当于复制接口的所有成员。接口也是用关键字“extends”来继承。
// 一个interface可以同时继承多个interface，实现多个接口成员的合并。用逗号隔开要继承的接口。

interface Shape {
    color: string;
}

interface PenStroke {
    penWidth: number;
}

interface Square extends Shape, PenStroke {
    sideLength: number;
}


//  泛型即参数化类型，也就是说所操作的数据类型被指定为一个参数。这种参数类型可以用在类、接口和方法的创建中。
abstract class BaseType {
    BizType: string;
}
class ParamType extends BaseType {
    BizType: string = 'Po';
}
interface OI<TypeArg extends BaseType> {
    getContext(): TypeArg;
}
class AO<TypeArg extends BaseType> {
    Context: TypeArg;
    OpCode: string;

    constructor(OpCode: string, Context: TypeArg) {
        this.OpCode = OpCode;
        this.Context = Context;
    }
}
class PO extends AO<ParamType> implements OI<ParamType> {
    constructor(OpCode: string, Context: ParamType) {
        super(OpCode, Context);
    }

    getContext(): ParamType {
        return this.Context;
    }
}

//  命名空间，代码内部化
namespace frameWork {
    class Utils {
        static isNullOrEmtry(input: string): boolean {
            if (input === null || input === "") {
                return true;
            }
            return false;
        }
    }
    console.log(Utils.isNullOrEmtry('test')); //  正确
}
//console.log(frameWork.Utils.isNullOrEmtry('test')); //  错误，不能在外部调用命名空间内代码

//  复用模块，这样可以一个模块名能在全部文件中调用，并且可继承、可重复定义
// 使用 module 关键字来定义模块，并在末尾加花括号即可用； 用export 关键字使接口、类等成员对模块外可见。

module Validation {  //定义模块
    export interface StringValidator {  //声明接口对外部可以使用
        isAcceptable(s: string): boolean;
    }

    var lettersRegexp = /^[A-Za-z]+$/;
    var numberRegexp = /^[0-9]+$/;

    export class LettersOnlyValidator implements StringValidator { //声明类对外部可用
        isAcceptable(s: string) {
            return lettersRegexp.test(s);
        }
    }

    export class ZipCodeValidator implements StringValidator {
        isAcceptable(s: string) {
            return s.length === 5 && numberRegexp.test(s);
        }
    }
}

var strings = ['Hello', '98052', '101'];
var validators: { [s: string]: Validation.StringValidator; } = {};
validators['ZIP code'] = new Validation.ZipCodeValidator();  //使用模块中的类
validators['Letters only'] = new Validation.LettersOnlyValidator();
// 显示匹配结果
for (var i = 0; i < strings.length; i++) {
    for (var pName in validators) {
        document.write('"' + strings[i] + '" ' + (validators[pName].isAcceptable(strings[i]) ? ' matches ' : ' does not match ') + pName + "<br>"); // 使用方法
    }
}