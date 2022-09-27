// use test;    //  进入名称为`test`的数据库

db.collectionName.drop();  //  丢弃数据库
//  移除文档记录
db.collectionName.remove({
    // 搜索条件condition，为空时则清空数据库内容
});

db.collectionName
//  搜索
    .find({
        //    搜索条件condition
        name: {
            $regex: "holmes"
        }
        , sex: {
            $in: ["female", "elian"]
        }
        , age: {
            $gt: 17
            , $lt: 30
        }
        , job: {
            $nin: ["engineer"]
        }
        , gps: {
            $within: {
                $: [[50, 49], 100]
            }
        }
    }, {
        // 返回字段field
        _id: 0
        , name: 1
        , job
    }, {
        // 搜索选项option
    })
    //  排序
    .sort({
        //  以`updateTime`字段为降序（从最近到远）
        updateTime: -1
    })
    //  限制返回数量
    .limit(20)
    //  跳过数量
    .skip(10)
;

//  更新已有的文档，save()可在文档不存在时插入，存在时则更新
db.collectionName.update(
    //  搜索
    {
        salePrice: {
            $regex: "[0-9]"
        }
    }
    //  更新
    , {
        //  增减，只对数值类型有效
        $inc: {
            num: 1
        }
        //  对数组堆入新数据
        , $push: {
            images: "http://www.host.com.cn/images/url/filename.ext"
        },
        //  对数组进行避免重复的更新，可配合$each填充多条记录
        $addToSet: {
            originImages: {
                $each: ["http://www.host.com.cn//filename.ext", "http://www.host.com.cn//filename.ext", "someUrlElse"]
            }
        },
        $set: {
            //  根据数组的索引进行更新
            "images.0": {
                null
            },
            //  根据（从上一个参数搜索条件中查出的）数组类型的值进行更新
            "images.$": {
                null
            }
        }
    }/* , isInsert:Boolean, isMutily:Boolean */);

//  根据条件建立索引，提升查找效率，mongodb本身查找过的内容也会自动建立索引，每个集合最多有64个索引
db.collectionName.ensureIndex({
    createTime: -1
});

//  聚合管道，查找并改变输出
db.getCollection("collectionName").aggregate([
        {
            //  过滤
            $match: {
                isDelete: false
            },
            //  显示
            $project: {
                title: 1
                , author: 1
                , isbn: {
                    //  自定义字段
                    prefix: {
                        //  获取字符串字段
                        $substr: [
                            //  【字段名称，起始位置，字符数量】
                            "$isbn", 0, 3
                        ]
                    }
                }
                , lastName: "$author.last"
                , hasDiscount: {
                    //  判断，可跟if then语句
                    $cond: {
                        if: {
                            $gte: ["$discount", 0]
                        },
                        then: true
                        , else: false
                    }
                },
                hasHigh: {
                    //  并集布尔操作符，根据数组表达式返回
                    $and: [[null], [false], true, 1]
                },
                hasHigh2: {
                    //  非集布尔操作符，根据数组表达式返回
                    $or: [undefined, 0]
                },
                noHasHigh: {
                    //  取反布尔操作符，根据数组表达式返回
                    $not: []
                },
                hasSameImgUrl: {
                    //  等值布尔操作符，根据数组表达式返回。在数组表达式中，每项都为数组类型，对比项内部有是否相同值，项之间不进行对比。如`thumbImgs`自身内部有重复值则返回true，`thumbImgs`和`OriginImgs`之间不会进行对比
                    $setEquals: ["$thumbImgs", "$originImgs"]
                },
                sameImgUrls: {
                    //  对比若干个数组的项，返回相同值数组，忽略顺序
                    $setIntersection: [
                        "$thumbImgs", "$originImgs"
                    ]
                },
                noMoreSame: {
                    //  返回 合并重复项的数组，忽略顺序
                    $setUnion: ["$thumbImgs", "$originImgs"]
                },
                somethingMore: {
                    //  返回后者比前者多出的值数组，忽略顺序
                    $setDifference: ["$thumbImgs", "$originImgs"]
                }
            },
            //  分组
            $group: {
                //  该操作根据`_id`的异同进行分组，设置为null则合并为一条文档记录
                _id: "$name"
                , count: {
                    //  求和，步长为1
                    $sum: 1
                }
                , totalPrice: {
                    $sum: {
                        //  求乘积
                        $multiply: [
                            "$price"
                            , "quantity"
                        ]
                    }
                }
                , averageQuantity: {
                    // 求平均值
                    $avg: "$quantity"
                }
            }
        },
        {
            //  拆分，根据数组类型字段`sizes`拆分为多条文档记录
            $unwind: "$sizes"
        }
    ],
    {
        //  优先于$sort
        $limit: 10,
        $skip: 5,
        $sort: {
            age: -1
            , posts: 1
        }
    },
    {
        //  根据字段`author`，创建集合副本
        $out: "author"
    }
);

//  返回执行命令后返回的错误信息
db.runCommand({getLastError:1});