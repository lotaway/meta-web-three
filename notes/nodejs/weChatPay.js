// nodejs实现微信支付V3（二）公众号支付
const request = require('request');
exports.jsApiPay = (req, res) => {
    const {bookingNo, appid, attach, mch_id, nonce_str, total_fee, notify_url, openid, body, timeStamp} = req.query;
    const url = "https://api.mch.weixin.qq.com/pay/unifiedorder";
    var formData = `<xml>
        <appid>${appid}</appid>
        <attach>${attach}</attach>
        <body>${body}</body>
        <mch_id>${mch_id}</mch_id>
        <nonce_str>${nonce_str}</nonce_str>
        <notify_url>${notify_url}</notify_url>
        <openid>${openid}</openid
        <out_trade_no>${bookingNo}</out_trade_no>
        <spbill_create_ip></spbill_create_ip>
        <total_fee>${total_fee}</total_fee>
        <trade_type>JSAPI</trade_type>
        <sign>${paysignjsapi(appid, attach, body, mch_id, nonce_str, notify_url, openid, bookingNo, '', total_fee, 'JSAPI')}</sign>
        </xml>`;
    // 1.先通过POST请求微信支付的统一下单接口
    request({url: url, method: 'POST', body: formData}, function (err, response, body) {
        if (!err && response.statusCode === 200) {
            console.log(body);
            var prepay_id = getXMLNodeValue('prepay_id', body.toString("utf-8"));
            var tmp = prepay_id.split('[');
            var tmp1 = tmp[2].split(']');
            //签名
            var _paySignjs = paysignjs(appid, nonce_str, 'prepay_id=' + tmp1[0], 'MD5', timeStamp);
            res.render('jsapipay', {prepay_id: tmp1[0], _paySignjs: _paySignjs});
            //res.render('jsapipay',{rows:body});
            //res.redirect(tmp3[0]);
        }
    });
};

// 签名加密算法

function paysignjsapi(appid, attach, body, mch_id, nonce_str, notify_url, openid, out_trade_no, spbill_create_ip, total_fee, trade_type) {
    var ret = {
        appid: appid,
        attach: attach,
        body: body,
        mch_id: mch_id,
        nonce_str: nonce_str,
        notify_url: notify_url,
        openid: openid,
        out_trade_no: out_trade_no,
        spbill_create_ip: spbill_create_ip,
        total_fee: total_fee,
        trade_type: trade_type
    };
    var string = raw(ret);
    var key = _key;
    string = string + '&key=' + key;
    var crypto = require('crypto');
    return crypto.createHash('md5').update(string, 'utf8').digest('hex');
}

// 签名算法要注意大小写，里面有很多坑在

function paysignjs(appid, nonceStr, _package, signType, timeStamp) {
    var ret = {
        appId: appid, nonceStr: nonceStr, package: _package, signType: signType, timeStamp: timeStamp
    };
    var string = raw1(ret);
    var key = _key;
    string = string + '&key=' + key;
    console.log(string);
    var crypto = require('crypto');
    return crypto.createHash('md5').update(string, 'utf8').digest('hex');
}

// 签名时候的参数不需要转换为小写的

function raw1(args) {
    var keys = Object.keys(args);
    keys = keys.sort();
    var newArgs = {};
    keys.forEach(function (key) {
        newArgs[key] = args[key];
    });
    var string = '';
    for (var k in newArgs) {
        string += '&' + k + '=' + newArgs[k];
    }
    string = string.substr(1);
    return string;
}

// 解析XML

function getXMLNodeValue(node_name, xml) {
    var tmp = xml.split("<" + node_name + ">");
    var _tmp = tmp[1].split("</" + node_name + ">");
    return _tmp[0];
}

// 客户端直接通过WeixinJSBridge来实现弹出支付的窗口

function jsApiCall() {
    WeixinJSBridge.invoke('getBrandWCPayRequest', {
        "appId": "wxebs20ae8d978330b", //公众号名称，由商户传入
        "timeStamp": "1414211784", //时间戳，自1970年以来的秒数
        "nonceStr": "ibuaiVcKdpRxkhJA", //随机串
        "package": "prepay_id=<%=prepay_id%>", "signType": "MD5", //微信签名方式：
        "paySign": "<%=_paySignjs%>" //微信签名
    }, function (res) {
        WeixinJSBridge.log(res.err_msg);
        //alert(res.err_code+res.err_desc+res.err_msg);
        //判断支付返回的参数是否支付成功并跳转
    });
}

function callpay() {
    if (typeof WeixinJSBridge == "undefined") {
        if (document.addEventListener) {
            document.addEventListener('WeixinJSBridgeReady', jsApiCall, false);
        } else if (document.attachEvent) {
            document.attachEvent('WeixinJSBridgeReady', jsApiCall);
            document.attachEvent('onWeixinJSBridgeReady', jsApiCall);
        }
    } else {
        jsApiCall();
    }
}