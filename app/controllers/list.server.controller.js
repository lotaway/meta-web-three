/**
 * 商品列表和商品详情页
 */
let ServiceGoods = require("../provider/goods");

module.exports = {
    start: function (req, res, next) {
        let serviceGoods = new ServiceGoods()
            ,page_size = req.query.size || 10
            , page_start = (req.query.page | 1 - 1) * page_size;

        serviceGoods.getGoods(page_size, page_start, function (err, doc) {
            if (!err && typeof doc === "object") {
                //res.json(result);  //  直接转为JSON格式输出给浏览器
                res.render('list', {
                    fakerData: [
                        {
                            name: 'hah',
                            price: 288.00
                        },
                        {
                            name: 'waters',
                            price: 100000.00
                        }
                    ],
                    data: doc      //  返回的文档doc是对象格式？
                });
            }
            else {
                next(err);
            }
        });
    }
};