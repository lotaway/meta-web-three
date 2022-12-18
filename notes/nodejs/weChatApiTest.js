var request = require("request")
    , querystring = require("querystring")
;

var postData = querystring.stringify({
    "goodsInfo": {
        "name": "testgoodsname",
        "coverImgUrl": "yTbXu_xWLZ6rst2WkspDa2Ky0Rehux-Z52MlvlWTOxXlsBxsvm_ksWU24bwkmj8l",
        "priceType": 2,
        "price": "30.00",
        "price2": "50.00",
        "url": "pages/index/index?redirect=/mobi/cn/goods/458.html"
    }
});

/*request({
    "method": "GET",
    "url": "https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=APPID&secret=APPSECRET"
})*/

request({
    "method": "POST",
    "url": "https://api.weixin.qq.com/wxaapi/broadcast/goods/add?access_token=33_vm4-tnQYys4Q50Zj8l4Lt75WgM78gidBs3rFig7CKPq_bVfs8yOeUpr9J_fsP1C-z91kEvU_ik-FXjSJL9ifOOhqb_vvHc4-nBvWtv9nJblEPtDqrsF3BiZJs4h54x_U26CObPYtw6a4sCazGKTfAIABFO",
    "Content-Type": "application/json",
    "body": encodeUrl(postData)
}, function (err, response, body) {
    if (!err && response.statusCode === 200) {
        console.log("返回数据：" + body);
    } else {
        throw new Error(err);
    }
});