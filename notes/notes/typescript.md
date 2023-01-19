@[TOC](Typescript的进阶学习笔记-讲解三划线指令、内置方法类型、infer、协变、逆变和交叉类型)

# 声明文件

## *system.ts*文件，注意`namespace`不需要`export`导出

```typescript
namespace system {
    export declare type IP = string

    export function getIP() {
        return "It's ip";
    }
}
```

## 在*main.ts*文件中使用三划线指令标签引入

```typescript
/// <reference path="./system.ts"/>
const ip: system.IP = system.getIP();
```

# 类参数自动转成成员变量

## 在类的构造器可以用`private`修饰形参，该形参会自动升级为成员变量，当实例化时，实参也将自动赋值给成员变量

```typescript
class ClassRoom {
    manualName: string;

    //  注意_classNumber使用了private修饰符
    constructor(private readonly _classNumber: number, className?: string) {
        //  这里的第一个形参`_classNumber`添加了`private`修饰，已自动赋值给`this._classNumber`，无需手动赋值。
        //  this._classNumber
        //  而第二个形参`className`没有`private`修饰，实例化后无法用`this.className`获取，若需要可以手动赋值。
        this.manualName = className;
    }

    showClassInfo() {
        console.log(`已定义：${this._classNumber}`);
        console.log(`未定义：${this.className}`);  // 若没有手动赋值，将不存在
        console.log(`手动定义'this.name'：${this.manualName}`);
    }
}
```

# 关于内置的泛型讲解

## `NonNullable`排除`null`和`undefined`的可能性

```typescript
type OriginData = {
    id: number
    title: string
}
let data1: OriginData = null; //  可以将变量赋值为null，之后再重新赋值
data1 = {
    id: 1,
    title: "二次赋值"
};
type SQLData = NonNullable<OriginData>
let data2: SQLData = null;  //  这样会报错，不允许赋值null
```

## `Partial`会将传入的类型内含键全部从必需变为可选

```tepescript
type NecessaryData = {
    id: number
    title: string
}
type OptionalData = Partial<NecessaryData>
```

上面的代码相当于：

```typescript
type OptionalData = {
    id?: number
    title?: string
}
```

实际使用：

```typescript
let finalData: NecessaryData = {
    id: 1,
    title: "必需定义所有属性"
}
let inputData1: OptionalData = {
    // id不需要定义
    title: "只要定义部分属性即可"
}
```

## `Require`与`Partial`相反，会将出啊惹怒的类型内含键全部从可选变为必需

```typescript
type OptionalData = {
    id?: number
    title?: string
}
type NecessaryData = Require<OptionalData>
```

上面的代码相当于：

```typescript
type necessaryData = {
    id: number
    title: string
}
```

## `Pick`从指定的类型中挑选其中几个键作为新类

```typescript
type FullData = {
    id: number
    title: string
    image?: string[]
    updateTime: Date
}
type InitData = Pick<FullData, "title" | "image">
let inputData2: InitData = {
    title: "初始化数据不需要id",
    // image: []    //  保留原类型里的可选项
}
```

## `Omit`与`Pick`相反，从指定类型中排除键

```typescript
declare type ShowData2 = Omit<NecessaryData, "id">
let showData2: ShowData2 = {
    // id: "",  //  already exclude
    title: "",
    image: [],
    desc: ""
}
```

## `Extract`判断指定的类型中是否继承了要求的键，否则返回`Never`

```typescript
interface IData {
    id: number
}

interface IBlog {
    id: number
    title: string
}

type InsertData = Extract<IBlog, IData>;
let inputData3: InsertData = {
    id: 3,
    title: "检查通过"
};
```

上述是对一个接口类型或对象类型进行检查，另一种用法是对联合类型检查筛选

```typescript
type Level = number | string | (() => number);
type CLevel = number | Function
type TargetLevel = Extract<Level, CLevel>;
```

以上相当于 type TargetLevel = number | (() => number)

## `Exclude`与`Extract`相反，判断指定类型中是否没有继承要求的键，若继承了则返回`Never`

```typescript
interface SaveData {
    id: number
    updateTime: Date
}

interface InputData {
    title: string
    image: string[]
    desc: string
}

type ShowData = Exclude<InputData, SaveData>;
let inputData: ShowData = {
    title: "检查通过",
    image: [],
    desc: ""
}
```

上述是对接口或对象类型的检查，另一种用法是对联合类型检查筛选

```typescript
type Level = number | string | (() => number);
type HLevel = string | Function
type TargetLevel = Exclude<Level, HLevel>;
```

以上相当于type TargetLevel = number

## `Record`将指定的类型中所有的键类型变成第二个参数传入的类型

```typescript
enum TargetRecord {
    Pending,
    Success,
    Error
}

interface ToBeKey {
    id: number
    value: string
}

type StateInfo = Record<TargetRecord, ToBeKey>;
let stateInfo: StateInfo = {
    [TargetRecord.Pending]: {
        id: TargetRecord.Pending,
        value: ""
    },
    [TargetRecord.Success]: {
        id: TargetRecord.Success,
        value: ""
    },
    [TargetRecord.Error]: {
        id: TargetRecord.Error,
        value: ""
    }
}
```

将接口类型`ToBeKey`变成了`TargetRecord`所有键的类型

## `ReturnType`传入函数类型，获取该函数的返回值类型

```typescript
type HasReturnFn = (a: number, b: number, format: string) => number;
type Result1 = ReturnType<HasReturnFn>;
const hasReturnFn: HasReturnFn = (a, b) => {
    return a + b;
};
type Result2 = ReturnType<typeof hasReturnFn>;
// Result1, Result2为相同类型
const arr = [1, "ss", {id: 1}];
```

## `InstanceType`类似`ReturnType`，相比获取函数返回值，这里传入的是类类型，获取的是类实例类型

```typescript
type ClassInstance = InstanceType<typeof ClassRoom>;
const instance: ClassInstance = new ClassRoom(512, "the second room");
```

## `Parameters`传入函数类型，获取该函数的形参类型并转成数组类型输出

```typescript
type FnParamsArr = Parameters<HasReturnFn>;
const fnParamsArr: FnParamsArr = [1, 2, "number"];
```

## `ConstructorParameter`类似`Parameters`，传入类类型，获取类构造器的形参类型并转成数组输出

```typescript
type ClassParamsArr = ConstructorParameters<typeof ClassRoom>;
const classParamArr: ClassParamsArr = [511, "the first one"];
```

## `infer`推断出类型

只能在条件类型的`extends`子句中使用，`infer`得到的类型只能在`true`语句中使用, 即`X ? Y : Z`中的`X`中推断，在`Y`位置引用

```typescript
type InferArray<T> = T extends (infer U)[] ? U : never;
//  (infer U)[]推断T是否为数组类型，若是的话返回数组项类型
type I0 = InferArray<[number, string]>; // 是数组，返回所有项的类型：string | number
type I1 = InferArray<number[]>; // 是数组，返回项类型：number
type I2 = InferArray<number>;   // 不是数组，返回never
// const i2: I2 = 1;    //  error, typeof 1 not never
type InferFirst<T extends unknown[]> = T extends [infer First, ...infer _] ? First : never
// infer P获取第一个元素的类型存储为First，而...infer _获取的是其他所有元素数组类型存储为_;
type I3 = InferFirst<[3, 2, 1]>; // number, typeof 3
type InferLast<T extends unknown[]> = T extends [...infer _, infer Last] ? Last : never;
//  类型上一个，获取最后一个元素的类型存储为Last
type I4 = InferLast<[3, 2, 1]>; // number, typeof 1
type InferParameters<T extends Function> = T extends (...args: infer FnParams) => any ? FnParams : never;
// ...args代表的是函数参数组成的元组, infer FnParams代表的就是推断出来的这个函数参数组成的元组的类型
type I5 = InferParameters<((arg1: string, arg2: number) => void)>; // [string, number]
type InferReturnType<T extends Function> = T extends (...args: any) => infer ReturnType ? ReturnType : never;
// 类似前面推断参数，infer ReturnType代表的就是推断出来的函数的返回值类型
type I6 = InferReturnType<() => string>; // string
type InferPromise<T> = T extends Promise<infer ResultData> ? ResultData : never;
//  推断出Promise中的返回值类型
type I7 = InferPromise<Promise<string>>; // string
// 推断字符串字面量类型的第一个字符对应的字面量类型
type InferString<T extends string> = T extends `${infer First}${infer _}` ? First : [];
type I8 = InferString<"John">; // 推断出John第一个字符字面量是J
```

# 协变

接口中继承的超集可以协变成父级类型：

```typescript
interface Person {
    name: string
}

interface Student extends Person {
    classname: string
}

let p1: Person = {
    name: "mimi"
}
let s1: Student = {
    name: "xixi",
    classname: "初一"
}
s1 = p1 //  不允许将父级赋值给超集，因为父级缺少超集的必须有的属性类型
p1 = s1 //  允许将超集赋值给父级，类型依旧是父级，无法调用超集的内容
p1.classname //  不允许调用超集属性，因为父级类型没有该属性，即使已经将超集赋值给父级也不行
```

类型中如果属性类型能对应上，则不需要接口继承关系也能自动完成协变：

```typescript
type Person = {
    name: string
}
type Student = {
    name: string
    classname: string
}
let p1: Person = {
    name: "mimi"
}
let s1: Student = {
    name: "xixi",
    classname: "初一"
}
p1 = s1 //  允许，理由同之前一样，即使没有两个类型没有继承关系也可以自动降级
```

# 逆变

与变量类型和接口相反，参数类型是逆变的，无论协变和逆变都是为了最终的代码安全。

```typescript
type PersonIntroFn = (args: Person) => string
type StudentIntroFn = (args: Student) => string
let personIntro: PersonIntroFn = args => `Hello, I'm ${args.name}`
let studentIntro: StudentIntroFn = args => `Hello, I'm ${args.classname} student ${args.name}`
personIntro = studentIntro  //  不允许，因为studentIntro实际代码有可能会调用超集参数中的超集属性，而赋值给[类型是父级参数的方法]却必定是不能调用的
studentIntro = personIntro //   允许，因为personIntro实际代码只会调用父级参数中的父级属性，而赋值给[类型是超集参数的方法]依旧能满足调用要求
```

# 交叉类型

通过&或|组合类型变成交叉类型

```typescript
interface Teacher {
    name: string
    teach: string
}

interface Student {
    name: string
    learn: string
}

type Mixer1 = Teacher & Student
//  等同于 type Mixer1 = { name: string, teach: string, learn: string }
type Mixer2 = Teacher | Student
//  等同于 type Mixer2 = { name: string}
```

另一种比较复杂的理解方式，强制组合泛型参数变成交叉类型：

```typescript
type UnionToIntersction<U> = (U extends U ? (a: U) => any : never) extends (a: infer R) => any ? R : never
type Copy<T> = {
    [K in keyof T]: T[K]
}
type res = Copy<UnionToIntersction<{ a: 1 } | { b: 3 }>>
//  等同于 type res = { a: 1, b: 3 }
```