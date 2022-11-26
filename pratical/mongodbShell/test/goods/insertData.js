let documents = [];

for (let i = 0; i < 100; i++) {
    documents.push({
        name: "测试商品名称" + i,
        salePrice: (Math.random() * 1000).toFixed(2),
        num: parseInt(Math.random() * 100),
        images: [
            "/images/upload/201605/06/201605061145502273.jpg",
            "/images/upload/201605/06/201605061203440655.jpg",
            "/images/upload/201605/06/201605061203485227.jpg",
            "/images/upload/201605/05/201605051524248332.jpg"
        ],
        meta: {
            createTime: Date.now()
            , updateTime: Date.now()
        }
    });
}
db.goods.insert(documents);