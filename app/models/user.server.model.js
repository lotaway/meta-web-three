/**
 * 用户数据模型
 */
// mongoose 链接
var mongoose = require('mongoose'),
    bcrypt = require('bcrypt'),
    bcrypt_level = 10;

//  声明模型
var Schema = mongoose.Schema;

//  定义模型结构
var userSchema = new Schema({
    username: {
        type: String,
        //default: '匿名用户',
        unique: true
    },
    password: String,
    title: String,
    content: {
        type: String
    },
    time: {
        type: Date,
        default: Date.now
    },
    age: {
        type: Number
    },
    meta: {
        createTime: {
            type: Date,
            default: Date.now
        },
        updateTime: {
            type: Date,
            default: Date.now
        }
    }
});

userSchema.pre('save', function (next) {

    var user = this;

    if (this.isNew) this.meta.createTime = this.meta.updateTime = Date.now();
    else this.meta.updateTime = Date.now();

    bcrypt.genSalt(bcrypt_level, function (err, salt) {
        if (err) return next(err);
        bcrypt.hash(user.password, salt, function (err, hash) {
            if (err) return next(err);
            user.password = hash;
            next();
        });
    });
});

mongoose.model('users', userSchema);