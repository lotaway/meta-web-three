function start(req, res) {
    res.render('/testIndex',{
        page: {
            title: title
        },
        template: {
            Floder: './public'
        },
        home: {
            slides: []
        },
        categories: [
            {
                name: 'cate name'
            }
        ],
        goods_list: [
            {
                id: 1,
                thumbImg: 'images/public/img.png',
                salePrice: 192.00,
                marketPrice: 292.00,
                limit: null
            }
        ]
    })
}
exports.start = start;