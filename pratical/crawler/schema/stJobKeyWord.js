const mongoose = require("mongoose")
    , db = mongoose.connect("mongodb://127.0.0.1:27017/job")
;
let     //  记录关键字出现次数
    dataArr = require("../input/keyWord")
    //  主表结构
    , keyWordSchema = new mongoose.Schema({
        name: {
            default: "未知",
            type: String,
            trim: true,
        },
        word: {
            type: [Array, String],
            require: true,
            unique: true
        },
        count: {
            type: Number,
            default: 0
        },
        extra: {
            type: Object,
            require: true,
            createTime: {
                type: Date,
                default: Date.now(),
                index: true
            },
            updateTime: {
                type: Date,
                default: Date.now(),
                index: true
            }
        }

    })
;

//  预先处理，更新存储的时间等内容
keyWordSchema.pre("save", function (next, done) {
    this.extra = this.extra || {};
    if (this.isNew) {
        this.extra.createTime = Date.now();
    }
    this.extra.updateTime = this.extra.createTime;
    next();
    if (typeof done === "function") {
        done();
    }
});
let StJobModel = db.model("stJobKeyWord", keyWordSchema);

StJobModel.create(dataArr, function (err, result) {
    if (err) {
        return console.log("存储出错：" + err);
    }
    console.log("成功：" + result);
    db.close();
});


module.exports = {
    dataArr
    , StJobModel
};