class ClassRoom {
    name: string;

    //  private param will auto save as instance value, use this.[param-name] to read the value
    constructor(private readonly _classNumber: number, className?: string) {
        this.name = className;  //  param without private have to set manual
    }

    showClassInfo() {
        console.log(`The member value is defined by constructor private param: ${this._classNumber}`);
        // console.log(`You can't get it without private: ${this.className}`);  // NO EXIST!
        console.log(`Param without private only can receive by 'this.name': ${this.name}`);
    }
}

//  about <generic>

//  Partial, use to transfer every key from necessary to optional
declare type DataId = number;
declare type NecessaryData = {
    readonly id: DataId
    title: string
    image: string[]
    desc: string
};
declare type OptionalData = Partial<NecessaryData>;
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
//  exclude null and undefined result
declare type Result3 = NonNullable<keyof typeof inputData1>;
//  Pick, use to take only a few props
declare type InitData = Pick<NecessaryData, "id" | "title">;
let inputData2: InitData = {
    id: 2,
    title: "few props is fine"
};
//  Require<InitData> == NecessaryData
//  Extract, make sure include some props, or will be never
declare type InsertData = Extract<InitData, { id: DataId }>;
let inputData3: InsertData = {
    id: 3,
    title: "true to be set value"
};
//  Exclude, make sure don't include some props, or will be never
declare type ShowData = Exclude<NecessaryData, { isDone: boolean }>;
let inputData4: ShowData = {
    id: 4,
    title: "this is some data",
    image: [],
    desc: ""
}
//  Omit, like Exclude & Pick mix, but exclude some props then return
declare type ShowData2 = Omit<NecessaryData, "id">

//  Record, use a type as key type
enum State {
    Pending,
    Success,
    Error
}

declare type StateInfo = Record<State, OptionalData>;
let stateInfo: StateInfo = {
    [State.Pending]: {
        id: State.Pending
    },
    [State.Success]: {
        id: State.Success
    },
    [State.Error]: {
        id: State.Error,
        title: "this is Error"
    }
};
//  ReturnType, as same as name, get the return type of <some function type>
declare type HasReturnFn = (a: number, b: number, format: string) => number;
declare type Result1 = ReturnType<HasReturnFn>;
const hasReturnFn: HasReturnFn = (a, b) => {
    return a + b;
};
declare type Result2 = ReturnType<typeof hasReturnFn>;
// Result1, Result2 is same type.
const arr = [1, "ss", {id: 1}];
// Parameters, get the params type from a function type
declare type FnParamsArr = Parameters<HasReturnFn>;
const fnParamsArr: FnParamsArr = [1, 2, "number"];    //  all should be match the function params type.
//  ConstructorParameters, similar as Parameters, but use in class constructor.
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