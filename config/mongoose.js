/**
 * 数据库统一连接
 */
var mongoose = require('mongoose');
var config = require('./config.js');

module.exports = function () {
    var db = mongoose.connect(config.mongodb);

    require('../app/models/login.server.model');
    require('../app/models/user.server.model');
    require('../app/models/goods.server.model');

    return db;
};