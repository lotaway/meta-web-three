/**
 mongodb 命令行开启服务，进入mongodb安装目录下bin运行：
 mongod

 设置数据库位置、输出位置和安装服务：
 mongod.exe --logpath d:/mongodb/logs --logappend --dbpath d:/websoft/mongodb/data --directoryperdb --serviceName MongoDB -install

 启动/停止服务(或者可以在本地服务中操作)：
 net start MongoDB
 net stop MongoDB

 robmongod图形管理软件在mongod3.0以上时需要设置验证版本为3（默认为5），否则无法认证通过，
 在mongo shell中（非robmongo里）运行以下命令：
 > use admin
 成功提示：switched to db admin
 > var s = db.system.version.findOne({"_id":"authSchema"})
 > s.currentVersion = 3
 成功提示：3
 > db.system.version.save(s)
 成功提示：WriteResult({ "nMatched" : 1, "nUpserted" : 0, "nModified" : 1 })

 完成验证版本修改后，添加最高权限用户：
 db.createUser({user:'yourname',pwd:'yourpassword',roles:[{role:'root',db:'admin'}]})

 完成后重新启动mongod服务：
 db.shutDownServer() // 关闭服务，或者在运行服务的命令行里Ctrl+C终止命令，或者直接关闭命令行
 mongod --auth      //  开启服务并启用认证

 认证登陆：
 use admin
 db.auth('yourname','yourpassword')
 */

//  help()  查看方法列表
//  db.dataBaseName.help()  查看目标数据库可执行的方法


var mongoose = require('mongoose');

var db = mongoose.connect('mongodb://127.0.0.1:27017/test');    //  设置数据库连接，参数为数据库地址
//'mongodb:lotaway@lotaway//127.0.0.1:27017/test'  TODO 认证？name@password//url:host/database

db.connection.on('open', function () {
    console.log('数据库连接成功~~');
});

/*  这个也是成功后触发的回调？
 db.connection.on('connected', function () {
 console.log('数据库连接');
 });*/

db.connection.on('disconnected', function () {
    console.log('数据库连接丢失');
});

db.connection.on('error', function (error) {
    console.log('数据库连接失败：' + error);
});

process.on('SIGIHT', function () {
    mongoose.connect.close(function () {
        console.log('程序已结束，关闭数据库连接');
        process.exit(0);
    });
});

//  文件形式存储的数据库模型骨架(传统意义的表结构)
var testSchema = new mongoose.Schema({
    //  数据类型有 String字符串、Number数值、Date日期、数组、Boolean布尔、null、ObjectId、Mixed、内嵌文档，也支持Buffer？
    name: {
        //  设置类型和默认值
        type: String,
        default: 'unknow',
        //  以下为预定义方法
        set: function (data) {
            //      当骨架对应的模型进行数据库【设置】操作时，对输入的数据进行处理
            return data;
        },
        get: function (data) {
            //      当骨架对应的模型进行数据库【获取】操作时，对输出的数据进行处理
            return data;
        },
        trim: true, //  预定义的修饰符，去除字符串开头和结尾空格
        unique: true,    //  设置为唯一索引，除了增加查询速度，还可以检查值是否唯一
        //  以下为预定义验证器
        required: true, //  表示字段是必须的
        enum: ['Mike', 'Jim', 'Tom'],       //  表示字段必须为数组其中之一的值
        match: /name/g,  //  表示字段必须符合该表达式
        //      自定义合法性验证器
        validate: function (data) {
            return data.length >= 4;
        }
    },
    phone: Number,    //  可以直接设置类型
    age: {
        type: Number,
        Max: 200,  //  最大值
        min: 0  //  最小值
    },
    createTime: {
        type: Date,
        default: Date.now,
        index: true //  辅助索引，用于增加查询速度
    },
    updateTime: {
        type: Date,
        default: Date.now,
        index: true
    },
    books: [],
    sex: Boolean,
    valuable: null,
    _id: ObjectId,  // mongodb的id类型，12字节的BSON类型字符串。按照字节顺序，依次代表：4字节：UNIX时间戳；3字节：表示运行MongoDB的机器；2字节：表示生成此_id的进程；3字节：由一个随机数开始的计数器生成的值。区别于自增的id，可以用于分布式系统
    friends: [
        //  内嵌文档
        {
            name: String,
            age: Number
        }
    ]
});

//  向Schema中追加属性
testSchema.add({
    unknow: Mixed
});

//  设置虚拟属性，虚拟属性不会存储到数据库里，但可以用于输出
testSchema.virtual('nickname').get(function () {
//      当骨架对应的模型进行操作时，触发此方法赋值属性
    return '假面超人MonCaCa';
});

//  预处理中间件之一，例如赋值创建时间，每次修改都更新时间
testSchema.pre('save', /*/!*是否并行触发，可省略，默认false*!/true,*/ function (next, /*可以传入并行触发的函数*/done) {
    if (this.isNew) {
        this.createTime = this.updateTime = Date.now();
    }
    else {
        this.updateTime = Date.now();
    }
    next();
    done();
});
//  后置处理中间件之一，例如保存成功后
testSchema.post('save', function (next) {
    console.log('save success');
    next();
});

// mongoose 模型实例方法
testSchema.methods = {
    findByUsername: function (username, callback) {
        return this.model('user').find({username: username}, callback);
    },
    /*xx: function(yy,callback){
     return yy;
     }*/
};
//  追加写法1
testSchema.methods('findByUsername', function (username, callback) {
});
//  追加写法2
testSchema.methods.findByUsername = function (username, callback) {
};

// mongoose 模型静态方法，静态方法在模型上就能使用
testSchema.statics.findByTitle = function (title, callback) {
    return this.find({title: title}, callback);
};

//  创建模型，把Schema结构对应到数据库里的集合上，通过该模型或者模型实体可以操作数据库（增删改查）
var TestModel = db.model('yourCollectionName', testSchema);

//  创建模型实体
var testEntity = new TestModel({
    //  实际数据
    name: "Lenka",
    age: 36,
    sex: true
});

//  伪数据
var conditions = {
        username: 'emtity_demo_username',
        title: 'emtity_demo_title'
    },
    fields = {
        title: 1,
        content: 1,
        time: 1
    },
    option = {},
    saveData = {
        name: "Lenka",
        age: 36,
        sex: true
    },
    update = {
        $set: {
            age: 27,
            title: 'model_demo_title_update'
        }
    };


//  ================查询===================

// 基于模型静态方法的查询

//  查询条件为空时返回全部文档
//  *最常用的查询选项就是限制返回结果的数量(limit函数)、忽略一点数量的结果(skip函数)以及排序(sort函数)
option = {
    limit: 20,   //  限制最多查询N条记录
    skip: 10,   //   在返回结果中跳过M条记录
    sort: {age: -1} //  排序操作，多个键/值对，键代表要排序的键名，值代表排序的方向，1是升序，-1是降序
};
TestModel.find(/*查询条件*/conditions, /*要返回的字段*/fields, /*查询选项*/option, /*回调*/function (err, result) {
    if (err) {
        //  出错时打印错误数据
        console.log('error:' + err);
    }
    else {
        //  查询成功时打印结果
        console.log(result);
    }
    //关闭数据库链接
    db.close();
});

//  返回首先找到的单个文档
TestModel.findOne(conditions, fields, {}, function (error, result) {
    if (error) console.log(error);
    else console.log(result);
    db.close();
});

//  Model.findById()   通过唯一的ID查找

//  此乃自定义的静态方法，通过标题查找
TestModel.findByTitle('emtity_demo_title', function (error, result) {
    if (error) console.log(error);
    else console.log(result);
    db.close();
});


// 基于自定义实例方法的查询
var testEntity1 = new TestModel({});
testEntity1.findByUsername('model_demo_username', function (error, result) {
    if (error) console.log(error);
    else console.log(result);
    db.close();
});

testSchema.set('toJSON', {getters: true, virtual: true}); //  设置输出为JSON字符串时，会包含虚拟属性

console.log('testSchema attr to json:' + JSON.stringify(testEntity1));  //  输出JSON为字符串


//Query.populate(path, [select], [model], [match], [options])

//  ================增加===================

//  基于模型
TestModel.create(saveData, function (err, result) {
    if (err) console.log('err:' + err);
    else console.log(result);
    db.close();
});

//  基于实体，要保存的数据是创建实体时决定的
var testEntity2 = new TestModel(saveData);
testEntity2.save(function (err, result) {
    if (err) console.log('err:' + err);
    else console.log(result);
    db.close();
});


//  ================更新===================

TestModel.update(/*查询条件*/conditions, /*要修改的内容*/update, /*查询选项*/option, /*回调*/function (err) {
    if (err) console.log('err:' + err);
    else console.log('update!');
    db.close();
});

//Model.findOneAndUpdate()，Model.findByIdAndUpdate()


//  ================删除记录===================

TestModel.remove(/*删除条件*/conditions, function (error) {
    if (error) console.log(error);
    else console.log('delete!');
    //关闭数据库链接
    db.close();
});

//Model.findOneAndRemove()，Model.findByIdAndRemove()

/*
 操作符：
 "$lt"(小于) { age:{ $lt: 20,$gt: 10 } }
 "$lte"(小于等于)
 "$gt"(大于)
 "$gte"(大于等于)
 "$ne"(不等于)
 "$in"(可单值和多个值的匹配) { name:{ $in: ['cali','jim'] } }
 "$or"(查询多个键值的任意给定值)    {$or:[ { name: 'test4' } , { age: 27}]}
 "$exists"(表示是否存在的意思)"$all"     {name: {$exists: true}
 */



//  ===================联表存储和查询===================

var Userss = new mongoose.model('Userss', {
//    可以直接在模型里设置模式
    username: String
});
var News = new mongoose.model('News', {
    title: String,
    author: {
        type: mongoose.Schema.ObjectId,
        ref: 'Userss'   //  关联模型
    }
});
var userss = new Userss({
    username: 'Jame Carry'
});
var news = new News({
    title: 'this is a title',
    author: userss  //  指向一个实体
});

userss.save(function (err) {
    if (err) return next(err);
    news.save(function (err) {
        if (err) return next(err);

        //  联表查询的实际例子（前面都是设置和保存数据）
        news
            .findOne()
            .populate(/*[path]指定被填充字段*/'author'/*,[select]指定填充字段, [model]指定模型，默认使用模式的ref属性, [match]查询条件, [options]查询选项*/)
            .exec(function (err, doc) {
                if (err) {
                    return next(err)
                }
                console.log('result:' + doc);
            });

    });
});

//  双向填充 , 填充Post的poster和comments字段以及comments的commenter字段:
Post.find({title: 'post-by-aikin'})
    .populate('poster comments')
    .exec(function (err, docs) {

        var opts = [{
            path: 'comments.commenter',
            select: 'name',
            model: 'User'
        }];

        Post.populate(docs, opts, function (err, populatedDocs) {
            console.log(populatedDocs[0].poster.name);                  // aikin
            console.log(populatedDocs[0].comments[0].commenter.name);  // luna
        });
    });