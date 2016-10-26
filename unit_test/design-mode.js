/**
 * Created by lotaway on 2016/5/29.
 */
//单例模式：生成类的唯一实例，如创建一个唯一的遮罩层，给所有弹窗方法共用
var single1 = (function () {
    //  用一个变量保存函数返回值，这个变量在同一个对象里是唯一的，不同对象之间是独立不共享的，无论调用多少次single1，只在第一次赋值result，后续都是直接返回result
    var result;
    return function () {
        return result || (result = document.body.appendChild(document.createElement('div')));
    };
})();

//  桥接模式，通过拆分函数完成独立变化，如把上面单例模式例子拆分成一个抽象函数和实现函数，传入新的回调（代理模式）来完成任务
var single2 = function (proxy) {
    var result;
    return function () {
        return result || (result = proxy.apply(this, arguments));
    }
};
var bridge = single2(function () {
    return document.body.appendChild(document.createElement('div'));
});

//  代理模式，抽象化一个函数，通过调用其他函数完成后续动作（桥接模式/观察者模式？）
function KeyManage() {
    console.log('make a start')
}
function proxy() {
    console.log('keep going')
}
var keyMager = KeyManage();
keyMager.listen('change', proxy());


//  简单工厂模式，集合了多个类，因实例化类型在编译期无法确定。在实例化时才决定到底是创建xhr的实例, 还是jsonp的实例. 是由实例化决定的
var Ajax = function (options) {
    var p, xhr;
    this.o = {
        url: "",
        type: "POST",
        data: "",
        dataType: "json",
        async: false,
        success: "",
        error: ""
    };
    if (typeof options === "object")
        for (p in this.o) {
            if (options[p] != undefined) this.o[p] = options[p];
        }
    switch (this.o.dataType.toUpperCase()) {
        case "JSON":
            this.o.dataType = "application/json";
            break;
        case "TEXT":
            this.o.dataType = "html/text";
            break;
        case "XML":
            this.o.dataType = "html/xml";
            break;
        default:
            break;
    }
    xhr = (function () {
        if (typeof XMLHttpRequest == "undefined") {
            XMLHttpRequest = function () {
                try {
                    return new ActiveXObject("Msxml2.XMLHTTP.6.0");
                }
                catch (e) {
                }
                try {
                    return new ActiveXObject("Msxml2.XMLHTTP.3.0");
                }
                catch (e) {
                }
                try {
                    return new ActiveXObject("Msxml2.XMLHTTP");
                }
                catch (e) {
                }
            };
        }
        return new XMLHttpRequest();
    })();
    this.done = function (fn) {
        if (typeof fn === 'function') {
            this.o.success = fn;
        }
    };
    this.onreadystatechange = function () {
        if (this.o.type.toUpperCase() != "JSONP") {
            //0表示未初始化，1表示正在加载，2表示加载完毕，3表示正在交互，4表示完成
            if (this.readyState == 4) {
                var t;
                switch (this.o.dataType) {
                    //访问返回的数据通过两个属性完成，一个是responseText，用于保存文本字符串形式的数据。另一个是responseXML，用于保存Content-Type头部中指定为text/xml的数据，其实是一个DocumentFragment对象
                    case "application/json":
                        if (typeof this.responseText === "string")
                            t = JSON.parse(this.responseText);
                        t = this.responseText;
                        break;
                    case "html/text":
                        t = this.responseText;
                        break;
                    case "html/xml":
                        t = this.responseXML;
                        break;
                    default:
                        break;
                }
                //200表示数据全部接受完毕
                if (this.status == 200) this.o.success(t);
                else this.o.error(t);
            }
        }
    };
    this.start = function () {
        if (this.o.type.toUpperCase() == "JSONP") {
            var script = document.createElement('script');
            script.setAttribute('src', this.o.url + "?" + this.o.data + '&callback=' + this.o.success);
            document.getElementsByTagName('head')[0].appendChild(script);
        }
        else {
            this.open(this.o.type, this.o.url, this.o.async);
            //xmlHttpRequest.charset = "UTF-8";
            if (this.o.type.toLocaleUpperCase() == "POST") {
                this.setRequestHeader("Content-Type", this.o.dataType);
            }
            this.send(this.o.data);
        }
    };
    return xhr;
};
//  实例化为json类型
var ajaxRequest1 = Ajax({url: 'your_ajax_url', type: 'post'});
ajaxRequest1.start();
ajaxRequest1.done(fn);
//  实例化为jsonp类型
var ajaxRequest2 = Ajax({url: 'your_ajax_url', type: 'jsonp'});
ajaxRequest2.start();
ajaxRequest2.done(fn);

//  观察者模式，又称发布者/订阅者模式，常见于监听事件
//  实现事件
var Events = function () {
    var listen, obj, one, remove, trigger, __this;

    obj = {};
    __this = this;
    //监听事件key和
    listen = function (key, fn) {
        var stack, _ref;  //stack是盒子
        stack = (_ref = obj[key]) != null ? _ref : obj[key] = [];
        return stack.push(fn);
    };

    one = function (key, fn) {
        remove(key);
        return listen(key, fn);
    };

    remove = function (key) {
        var _ref;
        return (_ref = obj[key]) != null ? _ref.length = 0 : void 0;
    };

    trigger = function () {  //面试官打电话通知面试者
        var fn, stack, _i, _len, _ref, key;

        key = Array.prototype.shift.call(arguments);
        stack = (_ref = obj[key]) != null ? _ref : obj[key] = [];

        for (_i = 0, _len = stack.length; _i < _len; _i++) {
            fn = stack[_i];
            if (fn.apply(__this, arguments) === false) {
                return false;
            }
        }
    };

    return {
        listen: listen,
        one: one,
        remove: remove,
        trigger: trigger
    }
};
//订阅者
var adultTv = Events();
adultTv.listen('play', function (data) {
    alert("今天是谁的电影" + data.name);
});
//发布者
adultTv.trigger('play', {'name': '麻生希'});

//  适配器模式，其实就是把一些原本不匹配的内容通过转接的方式匹配起来，适用于多人工作或者环境转变时，无需重构代码
var $id = function (id) {       //  把自定义的方法$id改为jQuery的Dom方法
    return jQuery("#" + id);
};

//  外观模式，就是内部调用多个其他函数，自身只包装为高层接口，方便多次调用同一批底层函数（保留底层函数分离更有灵活性）
var stopEvent = function (e) {   //同时阻止事件默认行为和冒泡
    e.stopPropagation();
    e.preventDefault();
};

// 访问者模式，抽象一个函数作为访问者，符合条件的人都可以作为被访问者调用它，这种方法原生js已实现，那就是apply()和call()方法
//  访问者
var Visitor = {};
Visitor.push = function () {
    return Array.prototype.push.apply(arguments);
};
//  被访问者
var obj = {};
obj.push = Visitor.push;
obj.push('haha');
console.log(obj[0] + ' ' + obj.length);

//  策略模式，为减少大量的if/else语法，通过分离为多个独立算法返回值实现，如验证表单时，有非空、长度、禁用词等验证方法
var validata = function (options) {
    var o = {
        //notNull，maxLength等方法只需要统一的返回true或者false，来表示是否通过了验证。
        notNull: function (value) {
            return value !== '';
        },
        dirtyWords: function (value) {

        },
        maxLength: function (value, maxLen) {
            return value.length() > maxLen;
        }
    }, p;
    for (p in options) {
        if (typeof o[p] === 'function') {
            if (!o[p](this.value, options[p])) return false;
        }
    }
    return true;
};
validata({
    notNull: true,
    dirtyWords: ['fuck', 'shit'],
    maxLength: 30
}).apply(inputEleName);

//模版方法模式，就是在父类中声明原型方法，在子类中进行重写（就算没有预先定义也是可以直接写，意义？）
var Life = function () {
};
Life.prototype.init = function () {
    this.DNA复制();
    this.出生();
    this.成长();
    this.衰老();
    this.死亡();
};
Life.prototype.出生 = function () {
};
Life.prototype.成长 = function () {
};
Life.prototype.衰老 = function () {
};
Life.prototype.死亡 = function () {
};

var Mammal = function () {
};
Mammal.prototype = Life.prototype;
Mammal.prototope.出生 = function () {
    胎生()
};
Mammal.prototype.成长 = function () {
    //再留给子类去实现
};
Mammal.prototope.衰老 = function () {
    自由基的过氧化反应()
};
Life.prototype.死亡 = function () {
    //再留给子类去实现
};
//再实现一个Dog类
var Dog = function () {
};
//Dog继承自哺乳动物.
Dog.prototype = Mammal.prototype;
var dog = new Dog();
dog.init();

//  中介者模式，A,B,C都通过这个中介函数链接E,F,G，而不需要前后两者之间直接的关联（其实就是MVC中的Controller角色）
var model1 = Model.create(), model2 = Model.create();
var view1 = View.create(), view2 = View.create();
var controler1 = Controler.create(model1, view1, function () {
    view1.el.find('div').bind('click', function () {
        this.innerHTML = mode1.find('data');
    });
});
var controler2 = Controler.create(model2, view2, function () {
    view1.el.find('div').bind('click', function () {
        this.innerHTML = model2.find('data');
    });
});

//  迭代器模式，其实就是封装好一些迭代方法用于事务场合
//  封装
var ObjForEach = function (obj, fn) {
    for (var i in obj) {
        var c = obj[i];
        if (fn.call(c, i, c) === false) {
            return false;
        }
    }
};
//  使用
ObjForEach({"a": 1, "b": 2}, function (i, n) {
    console.log(i + ' is ' + n);
});

//  组合模式，又称部分-整体模式。好处是不确定对象数量的情况下可以自动完成所有工作，如jQuery的Dom选择器中无论有一个还是多个Dom对象，调用方法时都会自动帮这些对象完成调用，实际上也是一种高层封装，例子略

//  备忘录模式，实际上就是利用了查找变量时，若函数内没有该变量，会自动查找上一级函数的变量，此时上一级函数的变量仍存储有内容可供使用，常见于匿名回调函数调用上一级函数的变量
var Page = function () {
    var page = 1,
        cache = {},
        data;
    return function () {
        if (cache[page]) {
            data = cache[page];
            render(data);
        } else {
            Ajax.send('cgi.xx.com/xxx', function (data) {
                cache[page] = data;
                render(data);
            })
        }
    }
}();

//  职责链模式，一个对象A向另一个对象B发起请求，如果B不处理，可以把请求转给C，如果C不处理，又可以把请求转给D。一直到有一个对象愿意处理这个请求为止。Dom事件的冒泡捕获机制，例如点击事件发生时，会当前节点触发，并依次向最高父节点传递。

//  享元模式，主要用来减少程序所需的对象个数。例如无限下拉列表中，通过确定出现在可视区域中的实际行数，加载时无需创建更多<li>标签，而是利用已经’消失‘在可视区域外的<li>标签重复利用。

/*
 状态模式，主要可以用于这种场景：
 1 一个对象的行为取决于它的状态
 2 一个操作中含有庞大的条件分支语句
 */
//  为了集中管理这些状态和代码，引入一个状态类，内部记录当前状态和完成修改状态，外部可以获取当前状态和修改状态
var StateManager = function () {
    var currState = 'wait';
    var states = {
        jump: function (state) {
        },
        wait: function (state) {
        },
        attack: function (state) {
        },
        crouch: function (state) {
        },
        defense: function (state) {
            if (currState === 'jump') {
                return false;  //不成功，跳跃的时候不能防御
            }
            //do something;     //防御的真正逻辑代码, 为了防止状态类的代码过多, 应该把这些逻辑继续扔给真正的fight类来执行.
            currState = 'defense'; //  切换状态
        }
    };
    var changeState = function (state) {
        states[state] && states[state]();
    };
    return {
        changeState: changeState
    }
};
var stateManager = StateManager();
stateManager.changeState('defense');