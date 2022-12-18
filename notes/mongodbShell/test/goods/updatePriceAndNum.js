//  old field type `salePrice` and `num` is String, change it into Number.
db.goods.find().forEach(function (item) {
    let num = parseInt(Math.random() * 100)
        , salePrice = Number((Math.random() * 10000).toFixed(2))
    ;

    db.goods.update({_id: item._id}, {$set: {salePrice, num}});
});