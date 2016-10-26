var userSession = require('../../data/user_session');
var mongoose = require('mongoose');

module.exports = {
    start: function (req, res) {

        if (req.session.sign) {//检查用户是否已经登录
            console.log('welcome' + req.session.name + '欢迎你再次登录');//打印session.name的值
        } else {
            if (req.body.password == undefined || req.body.password != userSession[req.body.user].password || !user[req.body.user]) {
                res.redirect('/login');
            }
            else {
                req.session.sign = true;
                req.session.name = '用户名哦';
                console.log('欢迎登陆！');
            }
        }

        res.render('user', {
            title: '一个人的世界，一飞冲天吧'
        });

        return ("Request handler 'start' was called.");

    },
//  登陆页 需查询数据库
    loginIn: function (req, res) {
        /* var login = mongoose.model('users');
         var query = {name: req.body.name, password: req.body.password};
         (function () {
         //count返回集合中文档的数量，和 find 一样可以接收查询条件。query 表示查询的条件
         login.count(query, function (err, doc) {
         if (doc == 1) {
         console.log((query.name + ":login success!" + new Date()));
         res.render('user', {
         title: 'user-center'
         });
         }
         else {
         console.log(query.name + ':login failed' + new Date());
         res.redirect('/');
         }
         })
         }());*/
    },
//  注册页
    signIn: function (req, res, next) {

        res.render('register', {});

    },
//  注册提交
    signUp: function (req, res, next) {

        var name = req.body.username,
            password = req.body.password;

        res.json({
            status: 1000,
            msg: 'get that',
            name: name,
            password: password
        });
    }
};