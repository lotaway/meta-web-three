/// <reference path="./system.ts"/>
const ip: system.IP = system.getIP();

class ClassRoom {
    manualName: string;

    //  构造器形参可以用`private`修饰符，该参数将自动转化成成员变量
    constructor(private readonly _classNumber: number, className?: string) {
        //  这里的第一个形参`_classNumber`添加了`private`修饰，已自动赋值给`this._classNumber`，无需手动赋值。
        //  this._classNumber
        //  而第二个形参`className`没有`private`修饰，实例化后无法用`this.className`获取，若需要可以手动赋值。
        this.manualName = className;
    }

    showClassInfo() {
        console.log(`已定义：${this._classNumber}`);
        // console.log(`未定义：${this.className}`);  // 若没有手动赋值，将不存在
        console.log(`手动定义'this.name'：${this.manualName}`);
    }
}

//  关于泛型

//  Partial会将传递的类型内含键都从必需转成可选
declare type DataId = number;
declare type NecessaryData = {
    readonly id: DataId
    title: string
    image: string[]
    desc: string
};
declare type OptionalData = Partial<NecessaryData>;
// 上面的代码相当于：declare type OptionalData = { id? title? image? desc? }`
let finalData: NecessaryData = {
    id: 1,
    title: "need every prop",
    image: [],
    desc: ""
};
let inputData1: OptionalData = {
    // id?
    title: "just single prop"
    // image?
    // desc?
};
//  排除null和undefined的可能性
declare type Result3 = NonNullable<keyof typeof inputData1>
//  从指定的类型中挑选其中几个键作为新类
declare type InitData = Pick<NecessaryData, "id" | "title">
let inputData2: InitData = {
    id: 2,
    title: "few props is fine"
};
//  Extract判断指定的类型中是否继承了要求的键，否则返回Never
declare type InsertData = Extract<InitData, { id: DataId }>;
let inputData3: InsertData = {
    id: 3,
    title: "true to be set value"
};
//  Exclude与Extract相反，判断指定类型中是否没有继承要求的键，若继承了则返回Never
declare type ShowData = Exclude<NecessaryData, { isDone: boolean }>;
let inputData4: ShowData = {
    id: 4,
    title: "this is some data",
    image: [],
    desc: ""
}
//  Omit与Pick相反，从指定类型中排除键
declare type ShowData2 = Omit<NecessaryData, "id">
let showData2: ShowData2 = {
    // id: "",  //  already exclude
    title: "",
    image: [],
    desc: ""
}

//  Record，将指定的类型中所有的键变成第二个参数传入的类型
enum TargetRecord {
    Pending,
    Success,
    Error
}

interface ToBeKey {
    id: number
    value: string
}

declare type StateInfo = Record<TargetRecord, ToBeKey>;
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
};
//  ReturnType获取函数返回的类型
declare type HasReturnFn = (a: number, b: number, format: string) => number;
declare type Result1 = ReturnType<HasReturnFn>;
const hasReturnFn: HasReturnFn = (a, b) => {
    return a + b;
};
declare type Result2 = ReturnType<typeof hasReturnFn>;
// Result1, Result2为相同类型
const arr = [1, "ss", {id: 1}];
// Parameters将传入的函数的形参类型转成数组类型输出
declare type FnParamsArr = Parameters<HasReturnFn>;
const fnParamsArr: FnParamsArr = [1, 2, "number"];
//  ConstructorParameter类似Parameters，将传入的类构造器的形参类型转成数组输出
declare type ClassParamsArr = ConstructorParameters<typeof ClassRoom>;
const classParamArr: ClassParamsArr = [511, "the first one"];
declare type ClassInstance = InstanceType<typeof ClassRoom>;
const instance: ClassInstance = new ClassRoom(512, "the second room");

// infer只能在条件类型的 extends 子句中使用
// infer得到的类型只能在true语句中使用, 即X ? Y : Z中的X中推断，在Y位置引用
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