var Class = (function () {
    //http://www.cnblogs.com/yexiaochai/p/4901341.html
    function Subclass() { }

    //我们构建一个类可以传两个参数，第一个为需要继承的类，
    //如果没有的话就一定会有第二个对象，就是其原型属性以及方法，其中initialize为构造函数的入口
    function create() {

        //此处两个属性一个是被继承的类，一个为原型方法
        var parent = null;
        var properties = Array.prototype.slice.call(arguments);

        if (Object.isFunction(properties[0]))
            parent = properties.shift();

        //新建类，这个类最好会被返回，构造函数入口为initialize原型方法
        function Kclass() {
            this.initialize.apply(this, arguments);
        }

        //扩展klass类的“实例”对象（非原型），为其增加addMethods方法
        Object.extend(Kclass, Class.Methods);

        //为其指定父类，没有就为空
        Kclass.superclass = parent;

        //其子类集合（require情况下不一定准确）
        Kclass.subclasses = [];

        //如果存在父类就需要继承
        if (parent) {
            //新建一个空类用以继承，其存在的意义是不希望构造函数被执行
            //比如 klass.prototype = new parent;就会执行其initialize方法
            Subclass.prototype = parent.prototype;
            Kclass.prototype = new Subclass;
            parent.subclasses.push(Kclass);
        }

        //遍历对象（其实此处这样做意义不大，我们可以强制最多给两个参数）
        //注意，此处为一个难点，需要谨慎，进入addMethods
        for (var i = 0, length = properties.length; i < length; i++)
            Kclass.addMethods(properties[i]);

        if (!Kclass.prototype.initialize)
            Kclass.prototype.initialize = Prototype.emptyFunction;

        Kclass.prototype.constructor = Kclass;
        return Kclass;
    }

    /**
     由于作者考虑情况比较全面会想到这种情况
     var Klass = Class.create(parent,{},{});
     后面每一个对象的遍历都会执行这里的方法，我们平时需要将这里直接限定最多两个参数
     */
    function addMethods(source) {

        //当前类的父类原型链，前面被记录下来了
        var ancestor = this.superclass && this.superclass.prototype;

        //将当前对象的键值取出转换为数组
        var properties = Object.keys(source);

        //依次遍历各个属性，填充当前类（klass）原型链
        for (var i = 0, length = properties.length; i < length; i++) {

            //property为键，value为值，比如getName: function(){}的关系
            var property = properties[i], value = source[property];

            /****************
             这里有个难点，用于处理子类中具有和父类原型链同名的情况，仍然可以调用父类函数的方案（我这里只能说牛B）
             如果一个子类有一个参数叫做$super的话，这里就可以处理了，这里判断一个函数的参数使用了正则表达式，正如
             var argslist = /^\s*function\s*\(([^\(\)]*?)\)\s*?\{/i.exec(value.toString())[1].replace(/\s/i, '').split(',');
      ****************/
            if (ancestor && Object.isFunction(value) && value.argumentNames()[0] == "$super") {

                //将当前函数存下来
                var method = value;
                /****************
                 第一步：

                 这里是这段代码最难的地方，需要好好阅读，我们首先将里面一块单独提出
                 value = (function (m) {
        return function () { return ancestor[m].apply(this, arguments); };
        })(property)
                 这里很牛B的构建了一个闭包（将方法名传了进去），任何直接由其父类原型中取出了相关方法
                 然后内部返回了该函数，此时其实重写了value，value
                 ***这里***有一个特别需要注意的地方是，此处的apply方法不是固定写在class上的，是根据调用环境变化的，具体各位自己去理解了

                 第二步：
                 首先value被重新成其父类的调用了，此处可以简单理解为（仅仅为理解）$super=value
                 然后下面会调用wrap操作vaule将，我们本次方法进行操作
                 wrap: function (wrapper) {
          var __method = this;
          return function () {
            return wrapper.apply(this, [__method.bind(this)].concat($A(arguments)));
          }
        }
                 可以看出，他其实就是将第一个方法（value）作为了自己方法名的第一个参数了，后面的参数不必理会
                 ****************/
                value = (function (m) {
                    return function () { return ancestor[m].apply(this, arguments); };
                })(property).wrap(method);
            }
            //为其原型赋值
            this.prototype[property] = value;
        }
        return this;
    }

    return {
        create: create,
        Methods: {
            addMethods: addMethods
        }
    };
})();