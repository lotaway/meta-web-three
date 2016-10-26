/**
 * 商品列表模型
 */
var mongoose = require('mongoose');
var listSchema = new mongoose.Schema({
    name: String,
    price: Number,
    num: Number,
    images: [
        String
    ]
});
listSchema.methods = {
    /**
     * 获取商品单页数据
     * @param page_size 数据量
     * @param page_start 数据开始位置
     * @param callback 回调
     */
    getPage: function (page_size, page_start, callback) {
        var List = mongoose.model('List');
        List
            .find()
            .skip(page_start)
            .limit(page_size)
            .exec(function (err, doc) {
                callback(err, doc);
            });
    },
    //  商品详情
    getDetail: function () {

    }
};

mongoose.model('List', listSchema);