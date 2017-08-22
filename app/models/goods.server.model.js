/**
 * 商品列表模型
 */
let mongoose = require('mongoose')
    , listSchema = new mongoose.Schema({
    name: String,   //  商品名称
    salePrice:  {  //  商品价格
        type: Number,
        default: 0
    },
    num: {    //  库存数量
        type: Number,
        default: 0
    },
    images: [       //  图片地址
        String
    ],
    meta: {
        createTime: Date.now(),
        updateTime: Date.now()
    }
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