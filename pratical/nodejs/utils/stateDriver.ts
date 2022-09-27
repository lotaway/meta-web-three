/**
 * 实现类似vue的数据绑定，包括数据到模板、DOM输入到数据和事件绑定
 * */
class EventController {

    //  事件类型
    static eventType: {
        trigger: 0,
        delete: 1,
        drop: 2
    };
    //  是否所有事件都触发
    static triggerAll = false;
    //  事件存储
    _event: {
        [key: string]: Array<Function>
    };

    constructor() {

    }

    //  添加事件订阅
    add(key: string, callback: Function) {
        if (!this._event[key]) {
            this._event[key] = [];
        }
        this._event[key].push(callback);
    }

    //  触发指定事件订阅
    trigger(key: string, eventType: number = EventController.eventType.trigger) {
        if (this._event[key]) {
            this._event[key].forEach(callback => callback({eventType}));
        }
    }

    //  删除指定事件订阅
    delete(key: string) {
        if (EventController.triggerAll) {
            this.trigger(key, EventController.eventType.delete);
        }
        this._event[key] = [];
    }

    //  丢弃所有订阅
    drop() {
        if (EventController.triggerAll) {
            Object.keys(this._event).forEach(key => {
                this.trigger(key, EventController.eventType.drop);
            });
        }
        this._event = {};
    }

}

module M {

    interface Methods extends Object {
        []: Function
    }

    export abstract class Mvvm {
        vm: typeof Mvvm;
        data: object;
        methods: Methods;
        constructor: Function;
    }

}

//  框架类
class Mvvm extends M.Mvvm {

    constructor(options: object) {
        super();
        this.vm = this;
        this.data = options.data;
        this.methods = options.methods;
        const observer = new Observer(this.data)
            , w = new Watcher(this, "name", val => updateView(val))
            , compile = new Compile(options.el, this.vm)
        ;
    }


    /**
     * 根据对象属性路径，最终获取值
     * @param {Object} obj 对象
     * @param {String} path 路径
     * return 值
     */
    static parsePath(obj: object, path: string): object {
        let bailRE = /[^\w.$]/;

        if (bailRE.test(path)) {
            return
        }
        let segments = path.split('.');

        for (let i = 0; i < segments.length; i++) {
            if (!obj) {
                return
            }
            obj = obj[segments[i]];
        }

        return obj;
    }

    /**
     * vue的set方法，用于外部新增属性 Vue.$set(target, key, val)
     * @param {Object} target 数据
     * @param {String} key 属性
     * @param {*} val 值
     */
    static set(target: object | Array<any> | string, key, val) {
        if (typeof target === "string") {
            target = Mvvm.parsePath(target, key);
        } else if (Array.isArray(target)) {
            target.length = Math.max(target.length, key);
            target.splice(key, 1, val);
            return val;
        }
        if (typeof target !== "object") {
            target = {};
        }
        target[key] = val;
        if (target.hasOwnProperty(key)) {
            return val;
        }
        new Observer(target);

        return val;
    }

    $set(...params) {
        Mvvm.set(...params);
    }

    /**
     * 将数据拓展到vue的根，方便读取和设置
     */
    proxy(key) {
        Object.defineProperty(this, key, {
            enumerable: true,
            configurable: true,
            get: () => this.data[key],
            set: newVal => this.data[key] = newVal
        });
    }

}

//  实现编译（解析模板绑定数据、事件和输入）
class Compile {

    vm: Mvvm;
    el: keyof ElementTagNameMap[any];
    fragment;

    constructor(el, vm) {
        this.vm = vm;
        this.el = document.querySelector(el);
        this.fragment = null;
        this.init();
    }

    /**
     * model更新节点
     */
    static modelUpdater(node, val) {
        node.value = typeof val == "undefined" ? "" : val;
    }

    init() {
        if (this.el) {
            console.log(`this.el:${this.el}`);
            //  移除页面元素生成文档碎片
            this.fragment = this.nodeToFragment(this.el);
            //  编译文档碎片
            this.compileElement(this.fragment);
            this.el.appendChild(this.fragment);
        } else {
            console.log("节点对象不存在");
        }
    }

    nodeToFragment(el) {
        let fragment = document.createDocumentFragment()
            , child = el.firstChild
        ;

        console.log(`el:${el}`);
        while (child) {
            fragment.appendChild(child);
            child = el.firstChild;
        }

        return fragment;
    }

    compileElement(fragment) {
        let childNodes = fragment.childNodes;

        Reflect.apply(Array.prototype.slice, childNodes, [node => {
            let reg = /\{\{\s*((?:.|\n)+?)\s*\}\}/g
                , text = node.textContent
            ;

            if (this.isElementNode(node)) {
                this.compileAttr(node);
            } else if (this.isTextNode(node) && reg.test(text)) {
                reg.lastIndex = 0;
                this.compileText(node, reg.exec(text)[1]);
            }
            if (node.childNodes && node.childNodes.length) {
                this.compileElement(node);
            }
        }]);
    }

    /**
     * 编译属性
     */
    compileAttr(node) {
        Array.prototype.forEach.call(node.attributes, attr => {
            let attrName = attr.name;

            // 只对vue本身指令进行操作
            if (this.isDirective(attrName)) {
                let exp = attr.value;

                // v-on指令
                if (this.isOnDirective(attrName)) {
                    this.compileOn(node, this.vm, exp, attrName);
                }
                // v-bind指令
                if (this.isBindDirective(attrName)) {
                    this.compileBind(node, this.vm, exp, attrName);
                }
                // v-model
                else if (this.isModelDirective(attrName)) {
                    this.compileModel(node, this.vm, exp, attrName);
                }

                node.removeAttribute(attrName);
            }
        })
    }

    /**
     * 编译文档碎片节点文本，即对标记替换
     */
    compileText(node, exp) {
        // 初始化视图
        this.updateText(node, this.vm.data[exp]);
        // 添加一个订阅者到订阅器
        let w = new Watcher(this.vm, exp, val => this.updateText(node, val));
    }

    /**
     * 编译v-on指令
     */
    compileOn(node, vm, exp, attrName) {
        // @xxx v-on:xxx
        let onRE = /^@|^v-on:/
            , eventType = attrName.replace(onRE, "")
            , cb = vm.methods[exp]
        ;

        if (eventType && cb) {
            node.addEventListener(eventType, cb.bind(vm), false);
        }
    }

    /**
     * 编译v-bind指令
     */
    compileBind(node, vm, exp, attrName) {
        // :xxx v-bind:xxx
        let bindRE = /^:|^v-bind:/
            , attr = attrName.replace(bindRE, "")
            , val = vm.data[exp];

        node.setAttribute(attr, val);
    }

    /**
     * 编译v-model指令
     */
    compileModel(node, vm, exp, attrName) {
        let self = this
            , val = this.vm.data[exp];

        // 初始化视图
        Compile.modelUpdater(node, val);

        // 添加一个订阅者到订阅器
        new Watcher(this.vm, exp, value => Compile.modelUpdater(node, value));

        // 绑定input事件
        node.addEventListener("input", function (e) {
            let newVal = e.target.value;

            if (val === newVal) {
                return;
            }
            self.vm.data[exp] = newVal;
            // val = newVal;
        });
    }

    /**
     * 更新文档碎片相应的文本节点
     */
    updateText(node, val) {
        node.textContent = typeof val === "undefined" ? "" : val;
    }

    /**
     * 属性是否是vue指令，包括v-xxx:,:xxx,@xxx
     */
    isDirective(attrName) {
        return /^v-|^@|^:/.test(attrName);
    }

    /**
     * 属性是否是v-on指令
     */
    isOnDirective(attrName) {
        return /^v-on:|^@/.test(attrName);
    }

    /**
     * 属性是否是v-bind指令
     */
    isBindDirective(attrName) {
        return /^v-bind:|^:/.test(attrName);
    }

    /**
     * 属性是否是v-model指令
     */
    isModelDirective(attrName) {
        return /^v-model/.test(attrName);
    }

    /**
     * 判断元素节点
     */
    isElementNode(node) {
        return node.nodeType == 1;
    }

    /**
     * 判断文本节点
     */
    isTextNode(node) {
        return node.nodeType == 3;
    }

}

//  示例
const mvvm = new Mvvm({
    data: {
        el: "#demo",
        data: {
            name: "hello world",
            color: "red"
        },
        methods: {
            clickHandler() {
                alert("触发点击");
            }
        }
    }
});
/**
 * vue对每个数据属性创建单独的消息管理器并封闭，在get方法中处理订阅，通过定义全局对象并伪装获取来触发get方法，使得全局对象添加进订阅者里。
 * 关键点是每个属性可动态附加多个订阅者，独立触发更改事件
 */
const eventController = new EventController();

const SELECTOR: string = "#name";

// 实现数据监听和视图同步更新。

/**
 * 视图更新
 * @param newVal 新数据
 */
function updateView(newVal) {
    let $name = document.querySelector(SELECTOR);

    $name.innerHTML = newVal;
}

//  监听器
class Observer {

    data: object;
    dep: Dep;

    constructor(data: object) {
        this.data = data;
        this.walk(this.data);
    }

    // 遍历
    walk(data) {
        Object.keys(data).forEach(key => {
            let childData = data[key];

            this.defineReactive(data, key);
            if (Array.isArray(childData) || typeof childData === "object") {
                this.walk(childData);
            }
        });
    }

    //  监听
    defineReactive(data, key) {
        let val = data[key]
            , dep = new Dep()   //  每个属性对应单个管理器
        ;

        Object.defineProperty(data, key, {
            enumerable: true,
            configurable: true,
            get: () => {
                if (Dep.target) {
                    dep.addSub(Dep.target);
                    Dep.target = null;
                }

                return val;
            },
            set(newVal) {
                if (newVal === val) {
                    return;
                }
                val = newVal;
                console.log(`属性${key}被监听了，现在的值为：${val}`);
                //  通知
                dep.notify(newVal);
            }
        });
        Dep.target = dep;
        const v = data[key];    //  强制触发
    }

}

interface Sub extends Object {
    update: (data) => void
}

//  消息订阅管理器
class Dep {

    static target: Sub = null;

    subs: Array<Sub>;

    constructor() {

    }

    addSub(sub) {
        this.subs.push(sub);
        console.log(`this.subs:${this.subs}`);
    }

    notify(data) {
        this.subs.forEach(sub => sub.update(data));
    }

    update(newVal) {
        updateView(newVal);
    }

}

// 订阅者
class Watcher implements Sub {

    vm: Mvvm;
    exp: string;
    cb: (value: any, oldValue: any) => void;
    value: any;

    constructor(vm, exp, cb) {
        this.vm = vm;
        this.exp = exp;
        this.cb = cb;
        this.value = this.get();
    }

    get() {
        Dep.target = this;
        //  强制触发监听
        return this.vm.data[this.exp];
    }

    update() {
        this.run();
    }

    run() {
        let value = this.vm.data[this.exp]
            , oldVal = this.value
        ;

        if (value !== oldVal) {
            this.value = value;
            this.cb.call(this.vm, value, oldVal);
        }
    }

}