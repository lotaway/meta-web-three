/**
 * 设置微信配置
 * @param url {string} 链接地址
 * @param config {object} 配置项
 * @param callback {function} 回调
 */
function setWxConfig(url, config, callback) {

    function setConfig(returnConfig) {
        /*debug: false,
         appId: 'wx79e1f52f51ce05d0',
         timestamp: 1495942418,
         nonceStr: 'AF164D312E6C80561AC75DF13A9E153E21A2CF9207AA98F0',
         signature: '7434f8c5c8030b57af9f021f42c38a5068636da8',
         jsApiList: ['onMenuShareTimeline', 'onMenuShareAppMessage', 'onMenuShareQQ', 'onMenuShareWeibo', 'onMenuShareQZone']
         });*/
        // returnConfig.debug = true;
        returnConfig.jsApiList = config.jsApiList || ["getLocation", 'onMenuShareTimeline', "onMenuShareAppMessage"];
        wx.config(returnConfig);
        callback(returnConfig);
    }

    if (!config.signature) {
        getWechatOfficialAccountConfig("url=" + encodeURIComponent(url.split("#")), function (data) {
            setConfig(data);
        });
    } else {
        setConfig(config);
    }
}

/**
 * 获取微信公众号配置
 * @param data {string} 传递的参数
 * @param successCB {function} 成功回调
 * @param errorCB {function} 失败回调
 */
function getWechatOfficialAccountConfig(data, successCB, errorCB) {
    var delay = 1.5;
    $.ajax({
        type: "post",
        url: "/ashx/get_weixin_config.ashx",
        data: data,
        dataType: "JSON",
        success: function (data) {
            switch (data.status) {
                //  成功
                case 1000:
                    delete data.status;
                    delete data.msg;
                    successCB(data);
                    break;
                //    缺少url
                case 1001:
                    art.dialog.tips("微信配置获取链接地址错误", delay);
                    break;
                //    accessToken参数获取失败
                case 1002:
                    // art.dialog.tips("微信公众号配置错误", delay);
                    console.log("微信公众号配置错误");
                    break;
            }
        },
        error: function (e) {
            if (typeof errorCB === 'function') {
                errorCB(e);
            } else {
                // art.dialog.tips("未知错误", delay);
                console.log("未知错误");
            }
        }
    });
}

/**
 * 根据订单号获取接口路径
 * @param orderNo {string} 订单号
 * @return {string} 接口路径
 */
function getApiByOrderNo(orderNo) {
    var handler = null //  处理方法
        , prefix = "/ashx/cn/"  //  路径前缀
    ;
    switch (true) {
        //  普通
        case orderNo.match(new RegExp("^PM_")) !== null:
            handler = "order_pay.ashx";
            break;
        //  秒杀
        case orderNo.match(new RegExp("^MS_")) !== null:
            handler = "seckill_order_pay.ashx";
            break;
        //  团购订单支付
        case orderNo.match(new RegExp("^GB_")) !== null:
            handler = "group_buy_order_pay.ashx";
        default:
            break;
    }
    return prefix + handler;
}

/**
 * 微信支付，判断hybrid、android、ios与web端进行不同的处理
 * @param {String} orderNo 订单号（必须）
 * @param {Number} hdfWxpayValue 微信新旧类型（必须）
 * @param {Number} orderTotal 订单数额（可选）
 * @param {Number} alimoney 预存款充值数额（可选）
 * @param {Function} callback 回调
 * @param {Function} errorCB 错误回调
 */
function weChatPay(orderNo, hdfWxpayValue, orderTotal, alimoney, callback, errorCB) {
    var _orderTotal = orderTotal ? orderTotal : ""
        , _alimoney = alimoney ? alimoney : ""
    ;

    //  微信小程序内嵌网页环境
    MainController.weChatMiniProgram.checks()
        .done(function () {
            callback({
                isFinish: false,
                next: function (redirect) {
                    wx.miniProgram.navigateTo({
                        url: "../weChatPay/weChatPay?from=mweb&orderNo=" + orderNo + _orderTotal + _alimoney + "&redirect=" + redirect,
                        success: function (result) {
                            setTimeout(function () {
                                location.href = redirect;
                            }, 1000 * 60);
                        },
                        fail: function (err) {
                            errorCB(err);
                        }
                    });
                }
            });
        })
        .fail(function () {
            //  是否micronet 的app  2016版应用调用
            if (MainController.application.check()) {
                if (typeof remote === "object" && typeof remote.appPay === 'function') {
                    remote.appPay(orderNo, orderTotal.split("=")[1], 'wxpay', window.location.host, callback, errorCB);
                } else {
                    art.dialog.tips("微信支付调用失败：remote.appPay方法不存在", 1.5);
                }
            } else if (navigator.userAgent.indexOf('micronetapp') > -1) {
                //  旧应用调用
                $.ajax({
                    type: "post",
                    url: "/pay/weixinApp/AppRequestReturn.aspx",
                    data: "showwxpaytitle=1&orderNo=" + orderNo + _orderTotal + _alimoney,
                    dataType: "JSON",
                    success: function (data) {
                        if (navigator.userAgent.indexOf('android') > -1) {
                            //调用安卓app里的方法
                            androidmicronet.androidweixinpay(data.orderno, data.ordername, data.moneyamount);
                        } else if (navigator.userAgent.indexOf('iphone') > -1) {
                            document.location.hash = "objciphonemicronet://" + "orderpay:" + ":/" + data.orderno + ":/" + encodeURIComponent(data.ordername) + ":/" + data.moneyamount;
                        }
                    }
                });
            } else {
                showWaiting();
                location.href = "/pay" + (hdfWxpayValue == "1" ? "/weixinpay" : "/wxpay") + "/payRequest.aspx?showwxpaytitle=1&orderNo=" + orderNo + _orderTotal + _alimoney;
            }
        });
}

function showWaiting() {
    var $loadingCartCon = $('<div id="loadingCart" class="container-waiting"><img class="img" src="/images/public/images/o_loading.gif" /><span class="text">正努力加载中。。。。。。</span></div>').appendTo($("body"));
    var width = $(window).width();
    var height = $(window).height();
    var padding_h = "" + ((height - 30) / 2) + "px";
    $loadingCartCon.css({
        "top": 0,
        "left": 0,
        "position": "fixed",
        "text-align": "center",
        "z-index": "999",
        "width": width,
        "height": height,
        "padding-top": padding_h,
        "background-color": "#ffffff",
        "opacity": 0.8
    });
}

/**
 * 拼团订单提交成功后的回调
 * @param obj 响应的数据
 */
function spellOrderSubmitSuccessCB(obj) {
    var delay = 1.5    //  延迟时间
        , timer //  计时器
        , triggerStatus = false   //  触发状态
        , href = obj.data ? "/mobi/cn/spellorder/result/" + obj.data.orderNo + ".html" : ""  // 链接
    ;

    function successCB() {
        clearTimeout(timer);
        if (triggerStatus === false) {
            triggerStatus = true;
            location.href = href;
        }
    }

    if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
    try {
        switch (obj.status) {
            case -1001:
                art.dialog.tips("很抱歉，您购买的商品库存不足,请修改此商品数量！<br />" + obj.data.goodsName, delay);
                break;
            case -1002:
                art.dialog.tips("很抱歉，您购买的商品已经下架！<br />" + obj.data.goodsName, delay);
                break;

            case -1003:
                art.dialog.tips("很抱歉，您购买的商品会员价已更改！<br />" + obj.data.goodsName, delay);
                setTimeout(function () {
                    window.location.reload();
                }, delay * 1000);
                break;
            case -1004:
                art.dialog.tips("很抱歉，您的收货地区不受支持！<br />" + obj.data.RegionDisabledInfo, delay);
                break;
            case -1005:
                art.dialog.tips("很抱歉，订单提交失败", delay);
                break;
            case -1006:
                art.dialog.tips("收货信息错误", delay);
                break;
            case -1007:
                art.dialog.tips("购物车没有商品", delay);
            case -1008:
                art.dialog.tips("此商品的拼团活动已经结束", delay);
                setTimeout("window.location.href='/mobi/cn/spell/list/1.html" + location.search + "'", delay * 1000);
                break;
            case 10035:
                art.dialog.tips("抱歉,此团已满人,请开团或者选择其它的团!", 2.5);
                setTimeout("location.href='/mobi/cn/spell/" + obj.goodid + ".html'", 3 * 1000);
                break;
            case 10036:
                art.dialog.tips("抱歉,此团已结束,请开团或者选择其它的团!", 2.5);
                setTimeout("location.href='/mobi/cn/spell/" + obj.goodid + ".html'", 3 * 1000);
                break;
            case -1100:
                OpenIdCardBoard();
                break;
            case 1000:
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/async_send_message.ashx",
                    data: "orderNo=" + obj.data.orderNo + "&sendType=1",
                    global: false,
                    async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                    success: successCB
                });
                timer = setTimeout(function () {
                    successCB();
                }, delay * 1000);
                break;
            default:
                art.dialog.tips("很抱歉，订单提交失败！", delay);
                setTimeout(function () {
                    window.location.reload();
                }, delay * 1000);
                break;
        }
    } catch (e) {
        errorHandler(e);
    }
}


/**
 * 订单提交成功后的回调
 * @param obj 响应的数据
 */
function orderSubmitSuccessCB(obj) {
    var delay = 1.5    //  延迟时间
        , timer //  计时器
        , triggerStatus = false   //  触发状态
        , href = obj.data ? "/mobi/cn/order/result/" + obj.data.orderNo + ".html" : ""  // 链接
    ;

    function successCB() {
        clearTimeout(timer);
        if (triggerStatus === false) {
            triggerStatus = true;
            location.href = href;
        }
    }

    if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
    try {
        switch (obj.status) {
            case -1001:
                art.dialog.tips("很抱歉，您购买的商品库存不足,请返回购物车修改此商品数量！<br />" + obj.data.goodsName, delay);
                break;
            case -1002:
                art.dialog.tips("很抱歉，您购买的商品已经下架,请返回购物车删除此商品！<br />" + obj.data.goodsName, delay);
                break;

            case -1003:
                art.dialog.tips("很抱歉，您购买的商品会员价已更改,请刷新后再提交订单！<br />" + obj.data.goodsName, delay);
                setTimeout("window.location.href='/mobi/cn/order/submit.html" + location.search + "'", delay * 1000);
                break;
            case -1004:
                art.dialog.tips("很抱歉，您的收货地区不受支持！<br />" + obj.data.RegionDisabledInfo, delay);
                //var lurl = "window.location.href='/mobi/cn/order/submit.html" + location.search + "'";
                //setTimeout(lurl, delay*1000);
                break;
            case -1005:
                art.dialog.tips("很抱歉，订单提交失败", delay);
                break;
            case -1006:
                art.dialog.tips("收货信息错误", delay);
                break;
            case -1007:
                art.dialog.tips("购物车没有商品", delay);
            case -1008:
                art.dialog.tips("此商品的拼团活动已经结束", delay);
                setTimeout("window.location.href='/mobi/cn/spell/list/1.html" + location.search + "'", delay * 1000);
                break;
            case -1100:
                OpenIdCardBoard();
                break;
            case 1000:
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/async_send_message.ashx",
                    data: "orderNo=" + obj.data.orderNo + "&sendType=1",
                    global: false,
                    async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                    success: successCB
                });
                timer = setTimeout(function () {
                    successCB();
                }, delay * 1000);
                break;
            default:
                art.dialog.tips("很抱歉，订单提交失败！", delay);
                setTimeout("window.location.href='/mobi/cn/order/submit.html" + location.search + "'", delay * 1000);
                break;
        }
    } catch (e) {
        errorHandler(e);
    }
}


/**
 * 秒杀订单提交成功后的回调
 * @param obj 响应的数据
 */
function seckillOrderSubmitSuccessCB(obj) {
    var delay = 1.5;    //  延迟时间
    if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
    try {
        switch (obj.status) {
            case -1001:
                art.dialog.tips("很抱歉，您购买的商品库存不足,请返回购物车修改此商品数量！<br />" + obj.data.goodsName, delay);
                break;
            case -1002:
                art.dialog.tips("很抱歉，您购买的商品已经下架,请返回购物车删除此商品！<br />" + obj.data.goodsName, delay);
                break;
            case -1003:
                art.dialog.tips("很抱歉，您购买的商品会员价已更改！<br />" + obj.data.goodsName, delay);
                setTimeout(function () {
                    window.location.reload();
                }, delay * 1000);
                break;
            case -1004:
                art.dialog.tips("很抱歉，您的收货地区不受支持！<br />" + obj.data.RegionDisabledInfo, delay);
                //var lurl = "window.location.href='/mobi/cn/order/submit.html" + location.search + "'";
                //setTimeout(lurl, delay*1000);
                break;
            case -1005:
                art.dialog.tips("很抱歉，订单提交失败", delay);
                break;
            case -1006:
                art.dialog.tips("收货信息错误", delay);
                break;
            case -1007:
                art.dialog.tips("购物车没有商品", delay);
                break;
            case -1100:
                OpenIdCardBoard();
                break;
            case 1000:
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/async_send_message.ashx",
                    data: "orderNo=" + obj.data.orderNo + "&sendType=1",
                    global: false,
                    async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                    success: function () {
                        location.href = "/mobi/cn/seckill/order/result/" + obj.data.orderNo + ".html";
                    }
                });
                setTimeout(function () {
                    location.href = "/mobi/cn/seckill/order/result/" + obj.data.orderNo + ".html";
                }, delay * 1000);
                break;
            default:
                art.dialog.tips("很抱歉，订单提交失败！", delay);
                setTimeout(function () {
                    window.location.reload();
                }, delay * 1000);
                break;

        }
    } catch (e) {
        errorHandler(e);
    }
}

/**
 * 根据时间段返回配送时间
 * @param start 开始时间
 * @param end 结束时间
 */
function getWeek(start, end) {
    start = start || $("#txtStartDate").val();
    end = end || $("#txtEndDate").val();
    var myDate = stringToDate(start),
        num = daysBetween(start, end) + 1,//计算有几天（时间段的长度）
        myArray = [];
    if (num > 0) {
        if (num < 7) {
            for (var i = 0; i < num; i++) {
                myArray.push((myDate.DateAdd('d', i)).getDay());
            }
        } else {
            myArray = ["0", "1", "2", "3", "4", "5", "6"];
        }
    }
    $.ajax({
        type: "POST",
        url: "/ashx/cn/delivery_time.ashx",
        data: "week=" + myArray,
        success: function (data) {
            $("#DeliveryTime").children().remove();
            $.each(data, function (i, o) {
                $("#DeliveryTime").append('<p><input type="checkbox" name="cbDeliveryTime" value="' + o.id + '">' + o.dayName + '</td><td>' + o.timeSlot + '</td><td>' + o.startTime + '-' + o.endTime + '</td><td><input id="cbIsDelivery' + o.id + '" type="checkbox" checked="checked">是否送货</p>');
            });
        }
    });
}

// 求两个时间的天数差 日期格式为 YYYY-MM-dd
function daysBetween(DateOne, DateTwo) {
    var OneMonth = DateOne.substring(5, DateOne.lastIndexOf('-'));
    var OneDay = DateOne.substring(DateOne.length, DateOne.lastIndexOf('-') + 1);
    var OneYear = DateOne.substring(0, DateOne.indexOf('-'));
    var TwoMonth = DateTwo.substring(5, DateTwo.lastIndexOf('-'));
    var TwoDay = DateTwo.substring(DateTwo.length, DateTwo.lastIndexOf('-') + 1);
    var TwoYear = DateTwo.substring(0, DateTwo.indexOf('-'));
    var cha = ((Date.parse(OneMonth + '/' + OneDay + '/' + OneYear) - Date.parse(TwoMonth + '/' + TwoDay + '/' + TwoYear)) / 86400000);
    return Math.abs(cha);
}

////日期计算(加几天)。本方法相当于重载（第一个参数是‘n、s、h、d、q……’，第二个参数是要加的长度）
Date.prototype.DateAdd = function (strInterval, Number) {
    var dtTmp = this;
    switch (strInterval) {
        case 's':
            return new Date(Date.parse(dtTmp) + (1000 * Number));
        case 'n':
            return new Date(Date.parse(dtTmp) + (60000 * Number));
        case 'h':
            return new Date(Date.parse(dtTmp) + (3600000 * Number));
        case 'd':
            return new Date(Date.parse(dtTmp) + (86400000 * Number));
        case 'w':
            return new Date(Date.parse(dtTmp) + ((86400000 * 7) * Number));
        case 'q':
            return new Date(dtTmp.getFullYear(), (dtTmp.getMonth()) + Number * 3, dtTmp.getDate(), dtTmp.getHours(), dtTmp.getMinutes(), dtTmp.getSeconds());
        case 'm':
            return new Date(dtTmp.getFullYear(), (dtTmp.getMonth()) + Number, dtTmp.getDate(), dtTmp.getHours(), dtTmp.getMinutes(), dtTmp.getSeconds());
        case 'y':
            return new Date((dtTmp.getFullYear() + Number), dtTmp.getMonth(), dtTmp.getDate(), dtTmp.getHours(), dtTmp.getMinutes(), dtTmp.getSeconds());
    }
};

function isDisplayUserAddress(region, obj) {
    var item = $("input[name='userAddress']:checked").val();
    $("#pUserAddress").html(obj);
    if (item != "0") {
        $("#addUserAddress").hide();
        var poststr = "region=" + encodeURI(region);

        $.ajax({
            type: "POST",
            data: poststr,
            dataType: "json",
            url: "/ashx/cn/order_info.ashx",
            success: function (obj) {
                var TabTitle = ".Settlement_Title",
                    TabCont = ".Settlement_Cont",
                    TabTitleStyle = "Settlement_Title1",
                    $TabParent;
                $("#liDelivery").html(obj);
                $TabParent = $(".Settlement_all > ul > li");
                $(TabTitle).click(function () {
                    if ($(this).next(TabCont).css("display") == "none") {
                        $TabParent.find(TabTitle).addClass(TabTitleStyle);
                        $TabParent.find(TabCont).slideUp(600);
                        $(this).removeClass(TabTitleStyle);
                        $(this).next(TabCont).slideDown(600);
                    }
                });
            }
        });
    } else {
        $("#addUserAddress").show();
        if ($("#selectProvinces option:selected").val() != "0") {
            poststr = "region=" + encodeURI($("#selectProvinces option:selected").text());

            $.ajax({
                type: "POST",
                data: poststr,
                dataType: "json",
                url: "/ashx/cn/order_info.ashx",
                success: function (obj) {
                    var TabTitle = ".Settlement_Title",
                        TabCont = ".Settlement_Cont",
                        TabTitleStyle = "Settlement_Title1",
                        $TabParent;
                    $("#liDelivery").html(obj);
                    $TabParent = $(".Settlement_all > ul > li");
                    $(TabTitle).click(function () {
                        if ($(this).next(TabCont).css("display") == "none") {
                            $TabParent.find(TabTitle).addClass(TabTitleStyle);
                            $TabParent.find(TabCont).slideUp(600);
                            $(this).removeClass(TabTitleStyle);
                            $(this).next(TabCont).slideDown(600);
                        }
                    });
                }
            });
        }
    }
}

function changePrice2money(s) {
    if (/[^0-9\.]/.test(s)) return "invalid value";
    s = s.replace(/^(\d*)$/, "$1.");
    s = (s + "00").replace(/(\d*\.\d\d)\d*/, "$1");
    s = s.replace(".", ",");
    var re = /(\d)(\d{3},)/;
    while (re.test(s))
        s = s.replace(re, "$1,$2");
    s = s.replace(/,(\d\d)$/, ".$1");
    return "¥" + s.replace(/^\./, "0.")
}

/****************************身份证输入的几个函数*********************************/
//弹窗初始化
function IdCardBoardInit() {
    $('.IdPopClose').click(function () {
        IdCardBoardClose();
    });
    $('.IdPopshaw').on('click', function () {
        IdCardBoardClose();
    });
}

//打开身份证输入弹窗
function OpenIdCardBoard() {
    $('.IdPopBox').show().after('<div class="IdPopshaw" style="height:' + 0 + 'px"></div>');
}

//关闭弹窗
function IdCardBoardClose() {
    $('#txtIdCard').val("");
    $('#txtTrueName').val("");
    $('.IdPopBox').hide();
    $('.IdPopshaw').remove();
    location.reload();
}

//身份证信息提交
function IdCardSubmit() {
    var txtIdCard = $("#txtIdCard2").val();
    var txtTrueName = $('#txtTrueName2').val();
    var dTime = new Date();
    $.ajax({
        type: "POST",
        url: "/cn/Home/IdCardSubmit/1.html",
        data: "idcard=" + txtIdCard + "&tn=" + txtTrueName + "&d=" + dTime.getTime(),
        success: function (json) {
            switch (json.ret) {
                case 1:
                    alert(json.msg);
                    $("#IdCardValue").val(txtIdCard.replace(/^(\d{4}).*(\d{2})$/g, "$1************$2"));
                    $("#TrueNameValue").val(txtTrueName.replace(/^(.{1}).*$/g, "$1**"));
                    IdCardBoardClose();
                    break;
                default:
                    alert(json.msg);
                    break;
            }
        }
    });
}

/**
 * 支付方式
 * @param pay_bank {String} 支付方式
 * @param hdfOrderNo {String} 订单号
 * @param extraParams {Object} 额外参数
 * @param orderTotal {String|Number} 订单金额
 * @param alimoney {String|Number} 预存款充值金额
 * @param errorCB {Function} 失败回调
 */
function payment(pay_bank, hdfOrderNo, extraParams, alimoney, orderTotal, errorCB) {
    if (orderTotal != null) orderTotal = "&orderTotal=" + orderTotal;
    else orderTotal = "";
    if (alimoney != null) alimoney = "&alimoney=" + alimoney;
    else alimoney = "";
    switch (true) {
        //  支付宝即时到账
        case pay_bank === payment.type.aLi:
            if (extraParams.hdfMicroMessenger === "1") {
                location.href = "/pay/m_alipay/weixin_default.aspx?orderNo=" + hdfOrderNo + orderTotal + alimoney;
            } else {
                location.href = "/pay/m_alipay/default.aspx?orderNo=" + hdfOrderNo + orderTotal + alimoney;
            }
            break;
        //  财付通即时到账
        case pay_bank === "tenPay":
            location.href = "/pay/m_tenPay/payRequest.aspx?orderNo=" + hdfOrderNo + orderTotal + alimoney;
            break;
        //  银联在线支付
        case pay_bank === "unionPay":
            location.href = "/pay/unionpay/micronetunionpay.aspx?orderNo=" + hdfOrderNo + orderTotal + alimoney;
            break;
        //  京东支付(网关支付)
        case /^jdpayDirectBank/.test(pay_bank):
            location.href = "/pay/jdpay_directBank/Send.aspx?orderNo=" + hdfOrderNo + "&pay_bank=" + pay_bank + "&mobi=1";
            break;
        //  京东手机支付
        case pay_bank === "jdpayMobi":
            location.href = "/pay/m_jd/paySubmit.aspx?orderNo=" + hdfOrderNo + orderTotal + alimoney;
            break;
        case pay_bank === "jdpay20":
            location.href = "/pay/Jdpay2/PayStartForm1.aspx?note=mobi&orderNo=" + hdfOrderNo + orderTotal + alimoney;
            break;
        //  微信支付
        case pay_bank === payment.type.weChat:
            weChatPay(hdfOrderNo, extraParams.hdfWxpay, orderTotal, alimoney, function (next) {
                return payment.redirectByOrderNumber(hdfOrderNo, {
                    next: next
                });
            }, errorCB);
            break;
        default:
            errorCB();
            break;
    }
}

//  支付类型
payment.type = {
    aLi: "directPay",
    weChat: "weixinPay"
};

/**
 * 支付成功重定向
 * @param orderNumber {string} 订单号
 * @param options {object} 额外选项
 */
payment.redirectByOrderNumber = function (orderNumber, options) {
    var redirect = "";  //  重定向地址

    options = options || {};
    switch (true) {
        //  充值
        case orderNumber.indexOf("OP_") > -1:
            redirect = "/mobi/cn/member/expense/records_1.html";
        //  普通商品
        case orderNumber.indexOf("PM_") > -1:
            // redirect = "/mobi/cn/member/order/" + orderNumber + "_info.html";
            redirect = "/mobi/cn/pay_success.html?orderNo=" + orderNumber;
            break;
        //拼团商品
        case orderNumber.indexOf("PT_") > -1:
            redirect = "/mobi/cn/member/spellorder/" + orderNumber + "_info.html";
            break;
        //  伙拼
        case orderNumber.indexOf("BG_") > -1:
            redirect = "/mobi/cn/member/buy_gang/order/" + orderNumber + "_info.html";
            break;
        //  团购
        case orderNumber.indexOf("GB_") > -1:
            redirect = "/mobi/cn/member/bulk/order/" + orderNumber + "_info.html";
            break;
        //  秒杀
        case orderNumber.indexOf("MS_") > -1:
            redirect = "/mobi/cn/member/seckill/order/" + orderNumber + "_info.html";
            break;
        //  砍价
        case orderNumber.indexOf("BO_") > -1:
            redirect = "/mobi/cn/member/bargain/UserOrderList.html";
            break;
        //  预售
        case orderNumber.indexOf("YD_") > -1:
            redirect = "/mobi/cn/member/yushou/yuding/order-list.html";
            break;
        //  线下扫码
        case orderNumber.indexOf("SC_") > -1:
            redirect = "/mobi/scanCode/paySuccess";
            break;
        //  积分
        case orderNumber.match("^\\d+$") !== null:
            redirect = "/mobi/cn/member/convertibility/order/" + orderNumber + "_info.html";
            break;
        //  其他，进我的订单列表页
        default:
            redirect = "/mobi/cn/member/1";
            break;
    }
    typeof options.next === "function" ? options.next(redirect) : location.href = redirect;
};

/**
 * 选择销售规格
 * @param goodsEntitys {string} 商品实体标识
 * @param specifications {string} 商品属性主标识
 * @param specificationsValue {string}  商品所选属性独立标识
 * @return {boolean}
 */
function chooseSpecifications(goodsEntitys, specifications, specificationsValue) {
    var poststr;

    if (goodsEntitys != null) {
        poststr = "goodsEntitys=" + goodsEntitys + "&specifications=" + specifications + "&specificationsValue=" + specificationsValue + "&goods=" + $("#hdGoods").val() + "&SpecificationsList=" + $("#hdSpecificationsList").val() + "&SpecificationsValueList=" + $("#hdSpecificationsValueList").val();
    } else {
        //  在详情页以外的页面调用选择规格，缺少商品实体id，此时参数为（Null，商品类id，回调）
        poststr = "goods=" + specifications;
    }
    $.ajax({
        type: "POST",
        url: "/ashx/cn/specifications.ashx",
        data: poststr,
        dataType: 'json',
        success: function (obj) {
            var htm = ''
                , ChBoxPrHtml = ""
                , Inventory //  库存
                , hdGoodsEntitys
            ;

            try {
                if (typeof specificationsValue === 'function') {
                    specificationsValue();
                }
                //  商品无规格
                if (obj.status == 1001) {
                    Inventory = obj.goodsEntitys[0].Inventory;
                    hdGoodsEntitys = obj.goodsEntitys[0].Id;
                } else if (obj.status == 1000) {
                    $("#goodsArtno").html('货品编号：' + obj.data.artNo);
                    if (obj.data.marketPrice != -1 && obj.data.salePrice != -1) {
                        $("#strong-saleprice").html('￥' + obj.data.salePrice);
                        $("#strong-marketprice").html('市场价￥' + obj.data.marketPrice);
                        $("#span-commissionVal").html(obj.data.commissionVal);

                        if (obj.data.enableCashBack == 1 && parseInt(obj.data.cashBack) > 0 && parseInt(obj.data.cashBackCycle) > 0)
                            $(".ChBoxPr").html('￥' + obj.data.salePrice + " 返现:￥" + obj.data.cashBack + " 周期:" + obj.data.cashBackCycle + obj.data.cycleUnit);
                        else
                            $(".ChBoxPr").html('￥' + obj.data.salePrice);
                        $(".PopProPrice span").eq(1).html(obj.data.spInventory);
                    }
                    //  会员价相关
                    obj.data.userankPriceInfo.forEach(function (item) {
                        htm += '<p' + item.isCurrent == 1 ? ' class="rankprice">' : '>' + item.name + '：￥' + item.price + '</p>';
                    });
                    $("#divPrice").html(htm);
                    htm = '';
                    //  已选择
                    obj.data.selected.forEach(function (item) {
                        htm += '&nbsp;' + item;
                    });
                    //  $("#liChooseSpecifications .Cred").html("已选择：" + htm);
                    $("#liChooseSpecifications").html("已选择：" + htm);
                    //  $(".PopProAttr").eq(1).html("已选择： " + htm);
                    //库存
                    $("#spInventory").html(obj.data.spInventory);
                    Inventory = obj.data.spInventory;
                    hdGoodsEntitys = obj.data.id;
                    $("#hdSpecificationsValueList").val(obj.data.svList);
                    $("#txtCount").val(obj.data.spInventory > 0 ? 1 : 0);
                    //规格
                    htm = '';
                    obj.data.spectionsType.forEach(function (item) {
                        htm += '<dt>' + item.name + '</dt><dd>';
                        item.specifications.forEach(function (s) {
                            var cls = obj.data.selected.indexOf(s.SpecificationsValueName) >= 0 ? " cur" : "";
                            if (s.SpecificationsType == 0) {
                                htm += '<span class="AttrWord ' + cls + '" onclick="chooseSpecifications.call(document,\'' + s.GoodsEntitys + '\',\'' + s.Specifications + '\',\'' + s.SpecificationsValue + '\');">' + s.SpecificationsValueName + '</span> ';
                            } else {
                                htm += '<span class="AttrPic ' + cls + '" onclick="chooseSpecifications.call(document,\'' + s.GoodsEntitys + '\',\'' + s.Specifications + '\',\'' + s.SpecificationsValue + '\');"><img src="' + s.SpecificationsValueImage + '" /></span>';
                            }
                        });
                        htm += '</dd>';
                    });
                }
                $(".Maximum").text(Inventory);
                $("#hdGoodsEntitys").val(hdGoodsEntitys);
            } catch (e) {
                errorHandler(e);
            }
            $("#ChBoxAttr").html(htm);
        }
    });
    return false;
}


///选择销售规格-拼团
function chooseSpecifications_spell(ge, s, sv) {
    var poststr = "goodsEntitys=" + ge + "&specifications=" + s + "&specificationsValue=" + sv + "&goods=" + $("#hdGoods").val() + "&SpecificationsList=" + $("#hdSpecificationsList").val() + "&SpecificationsValueList=" + $("#hdSpecificationsValueList").val();

    $.ajax({
        type: "POST",
        url: "/ashx/cn/specifications.ashx",
        data: poststr,
        dataType: 'json',
        success: function (obj) {
            var htm = '';
            try {
                $("#goodsArtno").html('货品编号：' + obj.data.artNo);
                if (obj.data.spellPrice != -1 && obj.data.salePrice != -1) {
                    $("#strong-saleprice").html('￥' + obj.data.spellPrice);
                    $("#strong-marketprice").html('市场价￥' + obj.data.salePrice);
                    $("#span-commissionVal").html(obj.data.commissionVal);
                    $(".ChBoxPr").html('￥' + obj.data.spellPrice);
                    $(".PopProPrice span").eq(1).html(obj.data.spInventory);
                }
                $("#divPrice").html(htm);
                htm = '';
                //  已选择
                obj.data.selected.forEach(function (item) {
                    htm += '&nbsp;' + item;
                });
                //  $("#liChooseSpecifications .Cred").html("已选择：" + htm);
                $("#liChooseSpecifications").html("已选择：" + htm);
                //  $(".PopProAttr").eq(1).html("已选择： " + htm);
                //库存
                $("#spInventory").html(obj.data.spInventory);
                $(".Maximum").text(obj.data.spInventory);
                $("#hdGoodsEntitys").val(obj.data.id);
                $("#hdSpecificationsValueList").val(obj.data.svList);
                $("#txtCount").val(obj.data.spInventory > 0 ? 1 : 0);
                //规格
                htm = '';
                obj.data.spectionsType.forEach(function (item) {
                    htm += '<dt>' + item.name + '</dt><dd>';
                    item.specifications.forEach(function (s) {
                        var cls = obj.data.selected.indexOf(s.SpecificationsValueName) >= 0 ? " cur" : "";
                        if (s.SpecificationsType == 0) {
                            htm += '<span class="AttrWord ' + cls + '" onclick="chooseSpecifications_spell(\'' + s.GoodsEntitys + '\',\'' + s.Specifications + '\',\'' + s.SpecificationsValue + '\');">' + s.SpecificationsValueName + '</span> ';
                        } else {
                            htm += '<span class="AttrPic ' + cls + '" onclick="chooseSpecifications_spell(\'' + s.GoodsEntitys + '\',\'' + s.Specifications + '\',\'' + s.SpecificationsValue + '\');"><img src="' + s.SpecificationsValueImage + '" /></span>';
                        }
                    });
                    htm += '</dd>';
                });

            } catch (e) {
                errorHandler(e);
            }
            $("#ChBoxAttr").html(htm);
        }
    });
    return false;
}


///选择销售规格-秒杀
function chooseSpecificationsSeckill(ge, s, sv) {
    var poststr = "goodsEntitys=" + ge + "&specifications=" + s + "&specificationsValue=" + sv + "&goods=" + $("#hdGoods").val() + "&SpecificationsList=" + $("#hdSpecificationsList").val() + "&SpecificationsValueList=" + $("#hdSpecificationsValueList").val();

    $.ajax({
        type: "POST",
        url: "/ashx/cn/specifications.ashx",
        data: poststr,
        dataType: 'json',
        success: function (obj) {
            var htm = '';
            try {
                $("#goodsArtno").html('货品编号：' + obj.data.artNo);
                if (obj.data.marketPrice != -1 && obj.data.salePrice != -1) {
                    $("#strong-saleprice").html('￥' + obj.data.salePrice);
                    $("#strong-marketprice").html('市场价￥' + obj.data.marketPrice);
                    $("#span-commissionVal").html(obj.data.commissionVal);


                    $(".ChBoxPr").html('￥' + obj.data.seckillPrice);
                    $(".PopProPrice span").eq(1).html(obj.data.spInventory);
                }
                //  会员价相关
                obj.data.userankPriceInfo.forEach(function (item) {
                    htm += '<p' + item.isCurrent == 1 ? ' class="rankprice">' : '>' + item.name + '：￥' + item.price + '</p>';
                });
                $("#divPrice").html(htm);
                htm = '';
                //  已选择
                obj.data.selected.forEach(function (item) {
                    htm += '&nbsp;' + item;
                });
                //  $("#liChooseSpecifications .Cred").html("已选择：" + htm);
                $("#liChooseSpecifications").html("已选择：" + htm);
                //  $(".PopProAttr").eq(1).html("已选择： " + htm);
                //库存
                $("#spInventory").html(obj.data.spInventory);
                $(".Maximum").text(obj.data.spInventory);
                $("#hdGoodsEntitys").val(obj.data.id);
                $("#hdSpecificationsValueList").val(obj.data.svList);
                $("#txtCount").val(obj.data.spInventory > 0 ? 1 : 0);
                //规格
                htm = '';
                obj.data.spectionsType.forEach(function (item) {
                    htm += '<dt>' + item.name + '</dt><dd>';
                    item.specifications.forEach(function (s) {
                        var cls = obj.data.selected.indexOf(s.SpecificationsValueName) >= 0 ? " cur" : "";
                        if (s.SpecificationsType == 0) {
                            htm += '<span class="AttrWord ' + cls + '" onclick="chooseSpecificationsSeckill(\'' + s.GoodsEntitys + '\',\'' + s.Specifications + '\',\'' + s.SpecificationsValue + '\');">' + s.SpecificationsValueName + '</span> ';
                        } else {
                            htm += '<span class="AttrPic ' + cls + '" onclick="chooseSpecificationsSeckill(\'' + s.GoodsEntitys + '\',\'' + s.Specifications + '\',\'' + s.SpecificationsValue + '\');"><img src="' + s.SpecificationsValueImage + '" /></span>';
                        }
                    });
                    htm += '</dd>';
                });

            } catch (e) {
                errorHandler(e);
            }
            $("#ChBoxAttr").html(htm);
        }
    });
    return false;
}


///加入购物车接口
var addShoppingCartApi = function (data, successCB) {
        $.ajax({
            type: "POST",
            url: "/ashx/cn/add_shopping_cart.ashx",
            data: data,
            dataType: 'json',
            success: successCB
        });
    }
    //  加入购物车预先检查
    , addToCartPreCheck = function (options) {
        var delay = 1.5
            , txtCountValue = Number(document.getElementById('txtCount').value) //  购买数量
        ;

        options = options || {};
        if ($("#isShowVisitorPrice").val() === "0") {
            window.location.href = "/mobi/cn/login.html?redirect=" + location.href;
            return false;
        }
        if (txtCountValue > 0) {
            addShoppingCartApi("goodsEntitys=" + document.getElementById('hdGoodsEntitys').value + "&count=" + txtCountValue, function (data) {
                addShoppingCartCB(data, options);
            });
        } else {
            art.dialog.tips("购买数量不能为" + txtCountValue + "！", delay);
            return false;
        }
    }
    //  加入购物车成功回调
    , addShoppingCartCB = function (data, options) {
        var delay = 1.5;

        options = options || {};
        switch (data.status) {
            case 1000:
                if (options.reload === false) {
                    $(".ChoiceBoxClose").trigger("click");
                } else {
                    setTimeout("window.location.reload()", delay * 1000);
                }
                PublicController.methods.updateCartAmount(data.sumAmount);
                //customButton('成功加入购物车！<br />购物车共' + data.sumAmount + ' 件商品，合计：￥' + data.sumTotal, '立即结算', '/mobi/cn/shopping/cart.html', '继续购物');
                art.dialog.tips("<div class='icon-success-add-cart'></div>商品已成功加入购物车~", delay);
                break;
            case 1001:
                window.location.href = '/mobi/cn/login.html';
                break;
            case 1002:
                art.dialog.tips("加入购物车失败，商品库存不足！", delay);
                break;
            case 1003:
                art.dialog.tips("添加购物车失败，商品超过限购数量！", delay);
                break;
        }
    };

$(function () {
    ///商品详情页 商品收藏/取消收藏
    $("#aFavorites").click(function () {
        var delay = 1.5;
        if ($(this).hasClass("disabled")) return false;
        $(this).addClass("disabled");
        if ($(".PsFun1").hasClass("PsFun1H")) {
            $.ajax({
                type: "POST",
                url: "/ashx/cn/goods_favorites.ashx",
                data: "type=cancel&id=" + $("#hdGoods").val(),
                dataType: 'json',
                success: function (data) {
                    switch (data.status) {
                        case 1000:
                            art.dialog.tips("商品取消收藏成功~", delay);
                            $(".PsFun1").removeClass("PsFun1H");
                            break;
                        case 1001:
                            art.dialog.tips("商品取消收藏失败！", delay);
                            break;
                    }
                }
            });
        } else {
            $.ajax({
                type: "POST",
                url: "/ashx/cn/goods_favorites.ashx",
                data: "id=" + $("#hdGoods").val(),
                dataType: 'json',
                success: function (obj) {
                    switch (obj.status) {
                        case 1000:
                            art.dialog.tips("商品收藏成功~", delay);
                            $(".PsFun1").addClass("PsFun1H");
                            break;
                        case 1001:
                            location.href = "/mobi/cn/login.html?redirect=" + location.href;
                            break;
                        case 1002:
                            art.dialog.tips("已收藏此商品", delay);
                            break;
                        case 1003:
                            art.dialog.tips("商品收藏失败！", delay);
                            break;
                    }
                }
            });
        }
        setTimeout("$('#aFavorites').removeClass('disabled')", delay * 1000);
        return false;
    });
    //  收藏页面 取消收藏
    if ($(".CollectDelBt").length) {
        $(".CollectDelBt").hammer().on('tap', function () {
            var delay = 1.5;
            $('.CollectCheck').each(function () {
                if ($(this).hasClass('cur')) {
                    var id = $(this).data('id');
                    if (id != '') {
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/goods_favorites.ashx",
                            data: "type=del&id=" + id,
                            dataType: 'json',
                            success: function (data) {
                                if (data.status == 1001) art.dialog.tips("删除收藏失败！", delay);
                                else {
                                    art.dialog.tips("删除收藏成功！", delay);
                                    setTimeout(function () {
                                        window.location.reload();
                                    }, delay * 1000);
                                }
                            }
                        });
                        return true;
                    }
                }
            });
        });
    }
    //  商品加入购物车
    $(".aAddShoppingCart").click(function () {
        addToCartPreCheck();

        return false;
    });
    //  套餐加入购物车
    $(".btnGoodsPackage").click(function () {
        addShoppingCartApi("goodsPackage=" + $(this).attr("data-value"), addShoppingCartCB);
        return false;
    });
    ///秒杀加入购物车
    $(".aAddShoppingCart_seckill").click(function () {
        var delay = 1.5;
        if ($("#isShowVisitorPrice").val() == 0) {
            window.location.href = "/mobi/cn/login.html?redirect=" + location.href;
            return false;
        }
        if (document.getElementById('txtCount').value == "0") {
            art.dialog.tips("请输入购买数量！", delay);
            return false;
        } else {
            $.ajax({
                type: "POST",
                url: "/ashx/cn/add_shopping_cart_seckill.ashx",
                data: "goodsEntitys=" + document.getElementById('hdGoodsEntitys').value + "&count=" + document.getElementById('txtCount').value,
                dataTyp: 'json',
                success: function (data) {
                    switch (data.status) {
                        case 1000:
                            // $("#shoppingCartAmount span").html(data.sumAmount);
                            //customButton('成功加入购物车！<br />购物车共' + data.sumAmount + ' 件商品，合计：￥' + data.sumTotal, '立即结算', '/mobi/cn/shopping/cart.html', '继续购物');
                            art.dialog.tips("<div class='icon-success-add-cart'></div>商品已成功加入购物车~", delay);
                            setTimeout("window.location.reload()", delay * 1000);
                            break;
                        case 1001:
                            //未登录
                            window.location.href = '/mobi/cn/login.html?redirect=' + location.href;
                            break;
                        case 1002:
                            art.dialog.tips("加入购物车失败，商品库存不足！", delay);
                            break;
                        case 1003:
                            art.dialog.tips("您已经买过" + data.buycount + "件了,给别的亲留一点吧", delay);
                            break;
                        case 10031:
                            art.dialog.tips("您买过" + data.buycount + "件,还可以买" + data.suggestBuy + "件", delay);
                            break;
                    }
                }
            });
        }
        return false;
    });
});


//////////////////////////////订单提交//////////////////////////////////////

$(function () {
    $("#selectProvinces").change(function () {
        if (typeof (disabledProvinceIDs) != "undefined" && disabledProvinceIDs.indexOf(',' + $("#selectProvinces").val() + ',') > -1) {
            $("#selectCity option").remove();
            $("#selectCity").append('<option value="0" selected="selected">未支持</option>');
            $("#selectCity").append('<option value="0" selected="selected">未支持</option>');
            $("#selectStreet option").remove();
            $("#selectStreet").append('<option value="0" selected="selected">未支持</option>');
            $("#selectStreet").append('<option value="0" selected="selected">未支持</option>');
            art.dialog.tips(typeof (RegionNotSupportWord_Province) != "undefined" ? RegionNotSupportWord_Province : "该省（地区）不受支持，请选择其它地区。", 1.5);
            //alert("该地区不受支持，请选择其它地区。")
            $("#spProvinces").html("<img src=\"/images/public/images/wrong.gif\" />");
        } else {
            $.ajax({
                type: "POST",
                url: "/ashx/cn/region.ashx",
                data: "pid=" + $("#selectProvinces").val(),
                success: function (data) {
                    if (data.length > 0) $("#spProvinces").html("");
                    else $("#spProvinces").html("<img src=\"/images/public/images/right.gif\" />");
                    if (typeof (disabledCityIDs) != "undefined") disabledCityIDs = ',';

                    $("#selectCity option").remove();
                    $("#selectCity").append('<option value="0" selected="selected">请选择...</option>');
                    $.each(data, function (index, item) {
                        $("#selectCity").append('<option value="' + item.id + '" >' + item.regionName + '</option>');
                        if (!item.bEnable && typeof (disabledCityIDs) != "undefined") disabledCityIDs += item.id + ',';
                    });
                    $("#selectStreet option").remove();
                    $("#selectStreet").append('<option value="0" selected="selected">请选择...</option>');
                }
            });
            if ($("#selectProvinces option:selected").val() != "0")
                var t = $("input[name='orderType']").val();
            var url = "/ashx/cn/order_info.ashx";
            if (t == "bulk") {
                url = "/ashx/cn/group_buy_order_info.ashx";
            } else if (t == "buygang") {
                url = "/ashx/cn/buy_gang_order_info.ashx";
            }
            $.ajax({
                type: "POST",
                url: url,
                data: "region=" + $("#selectProvinces option:selected").text(),
                success: function (obj) {
                    //多个
                    $("#delivery-div").children().remove();
                    for (var i = 0; i < obj.length; i++) {
                        var o = obj[i];
                        $("#delivery-div").append('<dd><div><div class="floatl">' + o.deliveryWay + ' +' + o.yunfei + '</div><div class="floatr"><input type="radio" name="deliveryWay" data-yunfei="' + o.yunfei + '" value="' + o.deliveryWay + ',' + o.yunfei + '" /></div></div></dd>');
                    }
                    var $TabParent = $(".Settlement_all > ul > li");
                    var TabTitle = ".Settlement_Title";
                    var TabCont = ".Settlement_Cont";
                    var TabTitleStyle = "Settlement_Title1";
                    $(TabTitle).click(function () {
                        var animateTime = 600;
                        if ($(this).next(TabCont).css("display") == "none") {
                            $TabParent.find(TabTitle).addClass(TabTitleStyle);
                            $TabParent.find(TabCont).slideUp(animateTime);
                            $(this).removeClass(TabTitleStyle);
                            $(this).next(TabCont).slideDown(animateTime);
                        }
                    });
                }
            });
        }
        return false;
    });
    //  选择城市
    $("#selectCity").change(function () {
        if (typeof (disabledCityIDs) != "undefined" && disabledCityIDs.indexOf(',' + $("#selectCity").val() + ',') > -1) {
            $("#selectStreet option").remove();
            $("#selectStreet").append('<option value="0" selected="selected">未支持</option>');
            $("#selectStreet").append('<option value="0" selected="selected">未支持</option>');
            art.dialog.tips(typeof (RegionNotSupportWord_City) != "undefined" ? RegionNotSupportWord_City : "该市（县/区）不受支持，请选择其它地区。", 1.5);
            //alert("该市不受支持，请选择其它地区。")
            $("#spProvinces").html("<img src=\"/images/public/images/wrong.gif\" />");
        } else {
            $.ajax({
                type: "GET",
                url: "/api/region",
                data: "pid=" + $("#selectCity").val(),
                success: function (data) {
                    if (data.length > 0) $("#spProvinces").html("");
                    else $("#spProvinces").html("<img src=\"/images/public/images/right.gif\" />");
                    if (typeof (disabledStreetIDs) != "undefined") disabledStreetIDs = ',';

                    $("#selectStreet option").remove();
                    $("#selectStreet").append('<option value="0" selected="selected">请选择...</option>');
                    $.each(data, function (index, item) {
                        $("#selectStreet").append('<option value="' + item.id + '" >' + item.regionName + '</option>');
                        if (!item.bEnable && typeof (disabledStreetIDs) != "undefined") disabledStreetIDs += item.id + ',';
                    });
                }
            });
        }
    });
    //  选择街道
    $("#selectStreet").change(function () {
        if (typeof (disabledStreetIDs) != "undefined" && disabledStreetIDs.indexOf(',' + $("#selectStreet").val() + ',') > -1) {
            art.dialog.tips(typeof (RegionNotSupportWord_Street) != "undefined" ? RegionNotSupportWord_Street : "该区（街道）不受支持，请选择其它地区。", 1.5);
            document.getElementById("selectStreet").selectedIndex = 0;
            $("#spProvinces").html("<img src=\"/images/public/images/wrong.gif\" />");
        } else {
            $("#spProvinces").html("<img src=\"/images/public/images/right.gif\" />");
        }

    });

    ////***************************订单销售网点地区筛选*******************************////
    $("#selectSoProvinces").change(function () {
        $.ajax({
            type: "GET",
            url: "/api/region",
            data: "pid=" + $("#selectSoProvinces").val(),
            success: function (data) {
                $("#selectSoCity option").remove();
                $("#selectSoCity").append('<option value="0" selected="selected">请选择...</option>');
                $.each(data, function (index, item) {
                    $("#selectSoCity").append('<option value="' + item.id + '" >' + item.regionName + '</option>');
                });
                $("#selectSoStreet option").remove();
                $("#selectSoStreet").append('<option value="0" selected="selected">请选择...</option>');
            }
        });
        if ($("#selectSoProvinces option:selected").val() != 0)
            $.ajax({
                type: "GET",
                url: "/api/salesOutlets",
                data: "province=" + $("#selectSoProvinces option:selected").text(),
                success: function (data) {
                    $("#selectSalesOutlets option").remove();
                    $("#selectSalesOutlets").append('<option value="0" selected="selected">请选择...</option>');
                    $.each(data, function (index, item) {
                        $("#selectSalesOutlets").append('<option value="' + item.id + '" >' + item.outletsTitle + '</option>');
                    });
                }
            });
        else
            $.ajax({
                type: "GET",
                url: "/api/salesOutlets",
                success: function (data) {
                    $("#selectSalesOutlets option").remove();
                    $("#selectSalesOutlets").append('<option value="0" selected="selected">请选择...</option>');
                    $.each(data, function (index, item) {
                        $("#selectSalesOutlets").append('<option value="' + item.id + '" >' + item.outletsTitle + '</option>');
                    });
                }
            });
        return false;
    });
    $("#selectSoCity").change(function () {
        $.ajax({
            type: "GET",
            url: "/api/region",
            data: "pid=" + $("#selectSoCity").val(),
            success: function (data) {
                $("#selectSoStreet option").remove();
                $("#selectSoStreet").append('<option value="0" selected="selected">请选择...</option>');
                $.each(data, function (index, item) {
                    $("#selectSoStreet").append('<option value="' + item.id + '" >' + item.regionName + '</option>');
                });
            }
        });
        $.ajax({
            type: "GET",
            url: "/api/salesOutlets",
            data: "province=" + $("#selectSoProvinces option:selected").text() + "&city=" + $("#selectSoCity option:selected").text(),
            success: function (data) {
                $("#selectSalesOutlets option").remove();
                $("#selectSalesOutlets").append('<option value="0" selected="selected">请选择...</option>');
                $.each(data, function (index, item) {
                    $("#selectSalesOutlets").append('<option value="' + item.id + '" >' + item.outletsTitle + '</option>');
                });
            }
        });
        return false;
    });

    $("#selectSoStreet").change(function () {
        $.ajax({
            type: "GET",
            url: "/api/salesOutlets",
            data: "province=" + $("#selectSoProvinces option:selected").text() + "&city=" + $("#selectSoCity option:selected").text() + "&street=" + $("#selectSoStreet option:selected").text(),
            success: function (data) {
                $("#selectSalesOutlets option").remove();
                $("#selectSalesOutlets").append('<option value="0" selected="selected">请选择...</option>');
                $.each(data, function (index, item) {
                    $("#selectSalesOutlets").append('<option value="' + item.id + '" >' + item.outletsTitle + '</option>');
                });
            }
        });
        return false;
    });

    $("input[name='pay_bank']").click(function () {
        if ($(this).prop("checked")) {
            $("#pPayMent").html($(this).attr("rel"));
        }
    });
});

//////////////////订单支付///////////////////////////
$(function () {
    $("#btnOrderPay").click(function () {
        var $self = $(this)
            , delay = 5
            , waitDelay = 30
            , orderNo = $("#hdfOrderNo").val()  //  订单号
            , orderTotal = $("#orderTotal").val()  //  订单金额
            , payBank = $("input[name=pay_bank]:checked")  //  支付方式对象
            , payBankName = payBank.attr("data-name")  //  支付方式名称
            , payUrl = getApiByOrderNo(orderNo)    //  支付路径
            , pay_bank
            , disabledText = "正在支付"
        ;

        if ($(this).hasClass("disabled")) {
            art.dialog.tips(disabledText, delay);
            return false;
        }
        if (payBank.length == 0) {
            $("input[name=pay_bank]").focus();
            art.dialog.tips("请选择支付方式！", delay);
            return false;
        }

        $(this).addClass("disabled").attr("data-text", $self.html()).html(disabledText);
        pay_bank = payBank.val();
        var integralParaStr = ($("#cbIntegral").length == 0) ? "" : "&cbInteralCheck=" + ($("#cbIntegral").is(":checked") ? 1 : 0) + "&orderIntegralPayAmount=" + $("#orderIntegralPayAmount").val() + "&userIntegralForOrder=" + $("#userIntegralForOrder").val();
        if ($("#cbPreDeposit").is(':checked') || ($("#cbIntegral").length == 1 && $("#cbIntegral").is(":checked") && payBank.val() != "预存款")) {
            $.ajax({
                type: "POST",
                url: payUrl,
                data: "orderNo=" + orderNo + "&pd=" + ($("#cbPreDeposit").is(":checked") ? 1 : 0) + integralParaStr + "&prepare=1",
                success: function (obj) {
                    var orderTotal = parseFloat(obj.data);
                    if (orderTotal >= 0) {
                        $.ajax({
                            type: "POST",
                            url: payUrl,
                            data: "orderNo=" + orderNo + "&payment=" + encodeURI("预存款") + integralParaStr,
                            success: function (data) {
                                $self.removeClass("disabled").html($self.attr("data-text"));
                                orderPaySuccessCB(data, orderNo, delay);
                            }
                        });
                    } else {
                        orderTotal = orderTotal * (-1);
                        payment(pay_bank, orderNo, {
                            hdfWxpay: $("#hdfWxpay").val(),
                            hdfMicroMessenger: $("#hdfMicroMessenger").val()
                        }, null, orderTotal, function (error) {
                            $self.removeClass("disabled").html($self.attr("data-text"));
                        });
                    }
                }
            });
            return false;
        } else {
            if (payBank.val() == "货到付款") {
                art.dialog.tips("正在提交数据，请稍候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", waitDelay);
                $.ajax({
                    type: "POST",
                    url: payUrl,
                    data: "orderNo=" + orderNo + "&payment=" + encodeURI("货到付款"),
                    success: function (msg) {
                        $self.removeClass("disabled").html($self.attr("data-text"));
                        if (msg.status == 1000) {
                            art.dialog.tips("恭喜您，订单处理成功！", delay);
                            setTimeout(function () {
                                location.href = "/mobi/cn/member/goods/order/" + orderNo + ".html";
                            }, delay * 1000);
                        }
                    }
                });
            } else if (payBank.val() == "预存款") {
                art.dialog.tips("正在提交数据，请稍候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", waitDelay);
                $.ajax({
                    type: "POST",
                    url: payUrl,
                    data: "orderNo=" + orderNo + "&payment=" + escape("预存款") + integralParaStr,
                    success: function (msg) {
                        if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                        $self.removeClass("disabled").html($self.attr("data-text"));
                        orderPaySuccessCB(msg, orderNo, delay);
                    }
                });
            } else {
                payment(payBank.val(), orderNo, {
                    hdfWxpay: $("#hdfWxpay").val(),
                    hdfMicroMessenger: $("#hdfMicroMessenger").val()
                }, null, orderTotal, function (error) {
                    art.dialog.tips(error, delay);
                    $self.removeClass("disabled").html($self.attr("data-text"));
                });
            }
        }
        return false;
    });

});


//////////////////拼团订单支付///////////////////////////
$(function () {
    $("#btnSpellOrderPay").click(function () {
        var $self = $(this)
            , delay = 5
            , waitDelay = 30
        ;
        if ($(this).hasClass("disabled")) {
            art.dialog.tips("正在处理支付", delay);
            return false;
        }
        var pay_bank;
        if ($("input[name=pay_bank]:checked").length == 0) {
            $("input[name=pay_bank]").focus();
            art.dialog.tips("请选择支付方式！", delay);
            return false;
        }
        $(this).addClass("disabled").attr("data-text", $self.html()).html("正在支付");
        pay_bank = $("input[name=pay_bank]:checked").val();
        if ($("#cbPreDeposit").is(':checked')) {//优先用预存款付款,如果预存款不够,先用第三方支付支付不够的部分
            $.ajax({
                type: "POST",
                url: "/ashx/cn/spell_order_pay.ashx",
                data: "orderNo=" + $("#hdfOrderNo").val() + "&pd=" + ($("#cbPreDeposit").is(":checked") ? 1 : 0) + "&prepare=1",
                success: function (obj) {
                    var orderTotal = parseFloat(obj.data);
                    if (orderTotal >= 0) {//预存款充足,直接用预存款支付所有金额
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/spell_order_pay.ashx",
                            data: "orderNo=" + $("#hdfOrderNo").val() + "&payment=" + encodeURI("预存款"),
                            success: function (data) {
                                $self.removeClass("disabled").html($self.attr("data-text"));
                                switch (data.status) {
                                    case 1000:
                                        art.dialog.tips("恭喜您，支付成功！", delay);
                                        $.ajax({
                                            type: "POST",
                                            url: "/ashx/cn/async_send_message.ashx",
                                            data: "orderNo=" + $("#hdfOrderNo").val() + "&sendType=2",
                                            global: false,
                                            async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                                            success: function () {
                                                location.href = "/mobi/cn/spell_pay_success.html?orderNo=" + $("#hdfOrderNo").val();
                                                //location.href = "/mobi/cn/member/goods/order/" + $("#hdfOrderNo").val() + ".html";
                                            }
                                        });
                                        setTimeout(function () {
                                            location.href = "/mobi/cn/spell_pay_success.html?orderNo=" + $("#hdfOrderNo").val();
                                            //location.href = "/mobi/cn/member/goods/order/" + $("#hdfOrderNo").val() + ".html";
                                        }, delay * 1000);
                                        break;
                                    case 1001:
                                        art.dialog.tips("您的余额不足以支付订单金额！<a href=\"/mobi/cn/member/pay/online.html\" style=\"color:Red; font-weight:bold;\">立即充值</a>", delay);
                                        break;
                                    case 1002:
                                        art.dialog.tips("你的订单已经支付，请不要重复支付！", delay);
                                        break;
                                    case 1003:
                                        art.dialog.tips("支付超时,订单已取消！", delay);
                                        setTimeout(function () {
                                            window.location.reload();
                                        }, 2000);
                                        break;
                                }
                            }
                        });
                    } else {//预存款不够,先用第三方支付支付不够的部分
                        orderTotal = orderTotal * (-1);
                        payment(pay_bank, $("#hdfOrderNo").val(), {
                            hdfWxpay: $("#hdfWxpay").val(),
                            hdfMicroMessenger: $("#hdfMicroMessenger").val()
                        }, null, orderTotal, function (error) {
                            $self.removeClass("disabled").html($self.attr("data-text"));
                        });
                    }
                }
            });
            return false;
        } else {
            if ($("input[name=pay_bank]:checked").val() == "预存款") {//只用预存款 支付
                art.dialog.tips("正在提交数据，请稍候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", waitDelay);
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/spell_order_pay.ashx",
                    data: "orderNo=" + $("#hdfOrderNo").val() + "&payment=" + escape("预存款"),
                    success: function (msg) {
                        if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                        $self.removeClass("disabled").html($self.attr("data-text"));
                        if (msg.status == 1001) {
                            art.dialog.tips("您的余额不足以支付订单金额！<a href=\"/mobi/cn/member/pay/online.html\" style=\"color:Red; font-weight:bold;\">立即充值</a>", delay);
                        }
                        if (msg.status == 1002) {
                            art.dialog.tips("您的订单已经支付，请不要重复支付！", delay);
                        }
                        if (msg.status == 1003) {
                            art.dialog.tips("支付超时,订单已取消！", delay);
                            setTimeout("parent.location.reload()", delay * 1000);
                        } else if (msg.status == 1000) {
                            art.dialog.tips("恭喜您，支付成功！", delay);
                            $.ajax({
                                type: "POST",
                                url: "/ashx/cn/async_send_message.ashx",
                                data: "orderNo=" + $("#hdfOrderNo").val() + "&sendType=2",
                                global: false,
                                async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                                success: function () {
                                    location.href = "/mobi/cn/spell_pay_success.html?orderNo=" + $("#hdfOrderNo").val();
                                    //location.href = "/mobi/cn/member/goods/order/" + $("#hdfOrderNo").val() + ".html";
                                }
                            });
                            setTimeout(function () {
                                location.href = "/mobi/cn/spell_pay_success.html?orderNo=" + $("#hdfOrderNo").val();
                                //location.href = "/mobi/cn/member/goods/order/" + $("#hdfOrderNo").val() + ".html";
                            }, delay * 1000);
                        }
                    }
                });
            } else {//只用第三方支付
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/order_status.ashx",
                    data: "orderNo=" + $("#hdfOrderNo").val(),
                    success: function (obj) {
                        switch (obj.status) {
                            case 1000:
                                //订单正常,进入支付流程
                                payment($("input[name=pay_bank]:checked").val(), $("#hdfOrderNo").val(), {
                                    hdfWxpay: $("#hdfWxpay").val(),
                                    hdfMicroMessenger: $("#hdfMicroMessenger").val()
                                }, null, $("#orderTotal").val(), function (error) {
                                    art.dialog.tips(error, delay);
                                    $self.removeClass("disabled").html($self.attr("data-text"));
                                });
                                break;
                            case 1001:
                                art.dialog.tips("支付超时,订单已取消！", 1.5);
                                setTimeout(function () {
                                    window.location.reload();
                                }, 2000);
                                break;
                            case 1003:
                                art.dialog.tips("订单已支付,请不要重复支付！", 1.5);
                                setTimeout(function () {
                                    window.location.reload();
                                }, 2000);
                                break;
                        }
                    }
                });
            }
        }
        return false;
    });
});


//////////////////////////////收货地址//////////////////////////////////////
$(function () {
    $("#formUserAddress").submit(function () {
        var delay = 1.5;
        if ($("#txtConsignee").val() == "") {
            $("#txtConsignee").focus();
            art.dialog.tips("请输入收货人姓名！", delay);
            return false;
        }
        if ($("#selectProvinces").val() == "0") {
            $("#selectProvinces").focus();
            art.dialog.tips("请选择省份/市！", delay);
            return false;
        }
        if ($("#selectCity") != null && $("#selectCity option").length > 1) {
            if ($("#selectCity").val() == "0") {
                $("#selectCity").focus();
                art.dialog.tips("请选择市(县/区)！", delay);
                return false;
            }
        }
        if ($("#selectStreet") != null && $("#selectStreet option").length > 1) {
            if ($("#selectStreet").val() == "0") {
                $("#selectStreet").focus();
                art.dialog.tips("请选择区/街道！", delay);
                return false;
            }
        }
        if ($("#txtAddress").val() == "") {
            $("#txtAddress").focus();
            art.dialog.tips("请输入详细地址！", delay);
            return false;
        }
        if ($("#txtZipcode").val() !== "" && !formatCheck($("#txtZipcode").val(), "zipCode")) {
            $("#txtZipcode").focus();
            art.dialog.tips("邮政编码格式不符！", delay);
            return false;
        }
        if ($("#txtTelephone").val() == "" && $("#txtMobile").val() == "") {
            $("#txtMobile").focus();
            art.dialog.tips("联系电话和手机号码至少填写一项！", delay);
            return false;
        }
        if ($("#txtMobile").val() !== "") {
            if (!formatCheck($("#txtMobile").val(), "mobile")) {
                $("#txtMobile").focus();
                art.dialog.tips("手机号码格式不符！", delay);
                return false;
            }
        }
        if ($("#txtTelephone").val() !== "") {
            if (!formatCheck($("#txtTelephone").val(), "telephone")) {
                $("#txtTelephone").focus();
                art.dialog.tips("电话号码格式不符！如：0754-88888888", delay);
                return false;
            }
        }
        var region = $("#selectProvinces option:selected").text();
        if ($("#selectCity option:selected").val() != "0")
            region += "," + $("#selectCity option:selected").text();
        if ($("#selectStreet option:selected").val() != "0")
            region += "," + $("#selectStreet option:selected").text();
        var isDefault;
        if ($("#isDefault").is(":checked")) {
            isDefault = 1;
        } else {
            isDefault = 0;
        }

        var uaId = $("#hdUserAddress").val()
            , redirect = getUrlParam("redirect") || PublicController.route.link.user.receivingAddressList
        ;

        PublicController.provider.addReceivingAddress({
            data: {
                consignee: $("#txtConsignee").val(),
                region: region,
                address: $("#txtAddress").val(),
                zipCode: $("#txtZipcode").val(),
                telephone: $("#txtTelephone").val(),
                mobile: $("#txtMobile").val(),
                isDefault: isDefault,
                id: uaId
            },
            success: function (obj) {
                if (obj.status == 1000) {
                    if ($.cookie('PresaleAddr') == 'yushouFillAddr') {
                        $.cookie('PresaleAddr', null, {path: '/'});
                    }
                    setTimeout("window.location='" + redirect + "'", delay * 1000);
                } else if (obj.status == 1002) {
                    setTimeout("window.location='" + redirect + "'", delay * 1000);
                }
            }
        });

        return false;
    });
});
//////////////////////////////用户信息//////////////////////////////////////
$(function () {
    var delay = 1.5;
    $("#formMemberInfo").submit(function () {
        var nickName = $("#txtNickName") //  昵称
        ;

        if (!formatCheck(nickName.val(), "nickname")) {
            nickName.focus();
            art.dialog.tips("昵称只能输入英文字母和数字！", delay);
            return false;
        }
        if ($("#txtEmail").val() === "") {

        } else if (!formatCheck($("#txtEmail").val(), "email")) {
            $("#txtEmail").select();
            art.dialog.tips("电子邮箱格式不符！", delay);
            return false;
        }
        if ($("input[name='sex']:checked").val() == null) {
            $("#radSex1").focus();
            art.dialog.tips("请选择性别！", delay);
            return false;
        }
        if ($("#txtTelephone").val() == "" && $("#txtMobile").val() == "") {
            $("#txtTelephone").focus();
            art.dialog.tips("联系电话和手机号码至少填写一项！", delay);
            return false;
        }
        if ($("#txtTelephone").val() != "") {
            if ($("#txtTelephone").val().match(/^(([0\+]\d{2,3}-)?(0\d{2,3})-)(\d{7,8})(-(\d{3,}))?$/) == null) {
                $("#txtTelephone").focus();
                art.dialog.tips("电话号码格式不符！如：0754-88888888！", delay);
                return false;
            }
        }
        if ($("#txtMobile").val() != "") {
            if (/^13\d{9}$/g.test($("#txtMobile").val()) || (/^15[0-35-9]\d{8}$/g.test($("#txtMobile").val())) || (/^17[0-35-9]\d{8}$/g.test($("#txtMobile").val())) || (/^18[0-9]\d{8}$/g.test($("#txtMobile").val()))) {
            } else {
                $("#txtMobile").focus();
                art.dialog.tips("手机号码格式不符！", delay);
                return false;
            }
        }
        if ($("#txtFax").val() != "") {
            if ($("#txtFax").val().match(/^(([0\+]\d{2,3}-)?(0\d{2,3})-)(\d{7,8})(-(\d{3,}))?$/) == null) {
                $("#txtFax").focus();
                art.dialog.tips("传真格式不符！", delay);
                return false;
            }
        }
        if ($("#selectProvinces").val() == "0") {
            $("#selectProvinces").focus();
            art.dialog.tips("请选择省份/市！", delay);
            return false;
        }
        if ($("#selectCity") != null && $("#selectCity option").length > 1) {
            if ($("#selectCity").val() == "0") {
                $("#selectCity").focus();
                art.dialog.tips("请选择市(县/区)！", delay);
                return false;
            }
        }
        if ($("#selectStreet") != null && $("#selectStreet option").length > 1) {
            if ($("#selectStreet").val() == "0") {
                $("#selectStreet").focus();
                art.dialog.tips("请选择区/街道！", delay);
                return false;
            }
        }
        var region = $("#selectProvinces option:selected").text();
        if ($("#selectCity option:selected").val() != "0")
            region += "," + $("#selectCity option:selected").text();
        if ($("#selectStreet option:selected").val() != "0")
            region += "," + $("#selectStreet option:selected").text();
        $.ajax({
            type: "POST",
            url: "/ashx/cn/user_info.ashx",
            data: "email=" + $("#txtEmail").val() + "&nickName=" + encodeURI(nickName.val()) + "&sex=" + $("input[name='sex']:checked").val() + "&telephone=" + $("#txtTelephone").val() + "&mobile=" + $("#txtMobile").val() + "&fax=" + $("#txtFax").val() + "&region=" + encodeURI(region) + "&address=" + $("#txtAddress").val() + "&qq=" + $("#txtQQ").val() + "&msn=" + $("#txtMSN").val() + "&wangwang=" + $("#txtWangwang").val() + "&remark=" + $("#txtRemark").val(),
            success: function (obj) {
                if (obj.status == 1000) {
                    art.dialog.tips("恭喜您，更新成功！", delay);
                    setTimeout("window.location.reload()", delay * 1000);
                } else if (obj.status == 1001) art.dialog.tips(obj.msg, delay);
            }
        });
        return false;
    });
});

//////////////////////////////商品订单//////////////////////////////////////
$(function () {
    $("#goodsOrderForm").submit(function () {
        var orderno = $("#txtOrderNo").val();
        if (orderno != '')
            window.location = "/mobi/cn/member/order/" + $("input[name='status']").val() + "/1/" + orderno + ".html";
        else
            window.location = "/mobi/cn/member/order/" + $("input[name='status']").val() + "/1.html";
        return false;
    });
});
//////////////////////////////团购商品订单//////////////////////////////////////
$(function () {
    $("#bulkOrderForm").submit(function () {
        var orderno = $("#txtOrderNo").val();
        if (orderno != '')
            window.location = "/mobi/cn/member/bulk/order/" + $("input[name='status']").val() + "/1/" + orderno + ".html";
        else
            window.location = "/mobi/cn/member/bulk/order/" + $("input[name='status']").val() + "/1.html";
        return false;
    });
});

function aConfirm(orderNo) {
    art.dialog({
        id: 'testID',
        content: '您确定已收到货物了吗？',
        lock: true,
        fixed: true,
        opacity: 0.1,
        button: [
            {
                name: '确定',
                callback: function () {
                    var url = "/ashx/cn/confirm_receiving.ashx";    //  普通商品退货接口地址
                    var data = "orderNo=" + orderNo;                //  普通商品传递的数据
                    //     云购处理
                    if (orderNo.indexOf("CFS_") > -1) {
                        url = "/ashx/CrowdfundGoods_shoppingHandle.ashx";
                        data = "acttype=confirmReceived&orderNo=" + orderNo;
                    }
                    art.dialog.tips("正在确认，请稍候  <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 60);
                    $.ajax({
                        type: "POST",
                        url: url,
                        data: data,
                        success: function (msg) {
                            if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                            if (msg.status == 1000 || msg == "1") {
                                window.location.reload();
                            } else {
                                art.dialog.tips('操作失败');
                            }
                        }
                    });
                    return false;
                },
                focus: true
            },
            {
                name: '取消',
                callback: function () {
                    art.dialog.tips('你取消了操作');
                }
            }
        ]
    });
}

//////////////////////////////商品订单详细//////////////////////////////////////
$(function () {
    $("input[name='pay_bank']").click(function () {
        if ($("input[name='pay_bank']:checked").val() == "预存款" || $("#PreDepositFun").val() == "False") {
            $("#pPd").hide();
        } else {
            $("#pPd").show();
        }
    });
    $("#aOrderInfoPay").click(function () {
        var order = $("#hdfOrderNo").val()
            , url = "/ashx/cn/order_pay.ashx"
            , $self = $(this)
            , delay = 5
        ;

        if ($("input[name='pay_bank']:checked").val() == null) {
            $("input[name='pay_bank']")[0].focus();
            art.dialog.tips("请选择支付方式！", delay);
            return false;
        }
        if (order.indexOf("GB_") >= 0) {
            url = "/ashx/cn/group_buy_order_pay.ashx";
        }
        if (order.indexOf("PT_") >= 0) {//拼团订单
            url = "/ashx/cn/spell_order_pay.ashx";
        }
        if (order.indexOf("MS_") >= 0) {//秒杀订单详细-->秒杀订单支付
            url = "/ashx/cn/seckill_order_pay.ashx";
        }
        // $(this).addClass("disabled").attr("data-text", $self.html()).html("正在支付");
        var integralParaStr = ($("#cbIntegral").length == 0) ? "" : "&cbInteralCheck=" + ($("#cbIntegral").is(":checked") ? 1 : 0) + "&orderIntegralPayAmount=" + $("#orderIntegralPayAmount").val() + "&userIntegralForOrder=" + $("#userIntegralForOrder").val();
        if (($("#cbPreDeposit").prop("checked") == "checked" || $("#cbPreDeposit").prop("checked") == true || ($("#cbIntegral").length == 1 && $("#cbIntegral").is(":checked"))) && $("input[name='pay_bank']:checked").val() != "预存款") {
            $.ajax({
                type: "POST",
                url: url,
                data: "orderNo=" + $("#hdfOrderNo").val() + "&pd=" + ($("#cbPreDeposit").is(":checked") ? 1 : 0) + integralParaStr + "&prepare=1",
                success: function (msg) {
                    //if (msg.status == 1001) {
                    //                        art.dialog.tips("您的余额不足以支付订单金额！<a href=\"/mobi/cn/member/pay/online.html\" style=\"color:Red; font-weight:bold;\">立即充值</a>", 1.5);
                    //                    } else
                    var orderTotal = parseFloat(msg.data);
                    if (msg.status == 1000) {
                        art.dialog.tips("正在提交数据，请稍候  <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
                        $.ajax({
                            type: "POST",
                            url: url,
                            data: "orderNo=" + $("#hdfOrderNo").val() + "&payment=" + encodeURI("预存款") + integralParaStr,
                            success: function (msg) {
                                if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                                //      $self.removeClass("disabled").html($self.attr("data-text"));
                                if (msg.status == 1001) {
                                    art.dialog.tips("您的余额不足以支付订单金额！<a href=\"/mobi/cn/member/pay/online.html\" style=\"color:Red; font-weight:bold;\">立即充值</a>", delay);
                                }
                                if (msg.status == 1002) {
                                    art.dialog.tips("您的订单已经支付，请不要重复支付！", delay);
                                }
                                if (msg.status == 1003) {
                                    art.dialog.tips("订单已取消,无法支付！", delay);
                                    setTimeout("parent.location.reload()", delay * 1000);
                                } else if (msg.status == 1000) {
                                    art.dialog.tips("恭喜您，支付成功！", delay);
                                    $.ajax({
                                        type: "POST",
                                        url: "/ashx/cn/async_send_message.ashx",
                                        data: "orderNo=" + $("#hdfOrderNo").val() + "&sendType=2",
                                        global: false,
                                        async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                                        success: function () {
                                            location.reload();
                                        }
                                    });
                                    setTimeout(function () {
                                        location.reload();
                                    }, delay * 1000);
                                }
                            }
                        });
                    } else {
                        orderTotal = orderTotal * (-1);
                        payment($("input[name='pay_bank']:checked").val(), $("#hdfOrderNo").val(), {
                            hdfWxpay: $("#hdfWxpay").val(),
                            hdfMicroMessenger: $("#hdfMicroMessenger").val()
                        }, null, orderTotal, function () {
                            //          $self.removeClass("disabled").html($self.attr("data-text"));
                        });
                    }
                }

            });
            return false;
        } else {
            payment($("input[name='pay_bank']:checked").val(), $("#hdfOrderNo").val(), {
                hdfWxpay: $("#hdfWxpay").val(),
                hdfMicroMessenger: $("#hdfMicroMessenger").val()
            }, null, $("#orderTotal").val(), function (error) {
                $self.removeClass("disabled").html($self.attr("data-text"));
            });

        }
        if ($("input[name='pay_bank']:checked").val() == "预存款") {
            art.dialog.tips("正在提交数据，请稍候  <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
            $.ajax({
                type: "POST",
                url: url,
                data: "orderNo=" + $("#hdfOrderNo").val() + "&payment=" + encodeURI("预存款") + integralParaStr,
                success: function (msg) {
                    if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                    if (msg.status == 1001) {
                        art.dialog.tips("您的余额不足以支付订单金额！<a href=\"/mobi/cn/member/pay/online.html\" style=\"color:Red; font-weight:bold;\">立即充值</a>", delay);
                    }
                    if (msg.status == 1002) {
                        art.dialog.tips("您的订单已经支付，请不要重复支付！", delay);
                    } else if (msg.status == 1000) {
                        art.dialog.tips("恭喜您，支付成功！", delay);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/async_send_message.ashx",
                            data: "orderNo=" + $("#hdfOrderNo").val() + "&sendType=2",
                            global: false,
                            async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                            success: function () {
                                window.location.reload();
                            }
                        });
                        setTimeout(function () {
                            window.location.reload();
                        }, delay * 1000);
                    }
                }
            });
        }
        return false;
    });
});


//////////////////////////////订单评论//////////////////////////////////////
$(function () {
    $("#orderCommentForm").submit(function () {
        var hasnopf = false,
            i,
            delay = 5,
            hiddens = $(".shop-rating input:hidden");
        if ($("#txtaContent").val() == "") {
            $("#txtaContent").focus();
            art.dialog.tips("请输入评论内容！", delay);
            return false;
        } else {
            if ($("#txtaContent").val().length > 500) {
                $("#txtaContent").select();
                art.dialog.tips("评论内容不能超过500个字符！", delay);
                return false;
            }
        }
        if ($("#txtasoContent").val() != undefined) {
            if ($("#txtasoContent").val() == "") {
                $("#txtasoContent").select();
                art.dialog.tips("请输入评论内容！", delay);
                return false;
            } else {
                if ($("#txtasoContent").val().length > 500) {
                    $("#txtasoContent").select();
                    art.dialog.tips("评论内容不能超过500个字符！", delay);
                    return false;
                }
            }
        }
        for (i = 0; i < hiddens.length; i++) {
            var val = hiddens.eq(i).val();
            if (val == "undefined" || val == '') {
                hasnopf = true;
                break;
            }
        }
        if (hasnopf) {
            art.dialog.tips("您还有没打分的选项！", delay);
            return false;
        }
        if ($("#txtCommentCode").val() == "") {
            $("#txtCommentCode").focus();
            art.dialog.tips("请输入验证码！", delay);
            return false;
        }
        $("#btnOrderComment").attr("disabled", true);
        $.ajax({
            type: "POST",
            url: "/ashx/cn/goods_comment.ashx",
            data: $("#orderCommentForm").serialize(),
            success: function (data) {
                $("#btnOrderComment").attr("disabled", false);
                if (data.status == 1003) {
                    window.location.href = "/mobi/cn/login.html";
                } else if (data.status == 1002) {
                    $("#txtCommentCode").select();
                    art.dialog.tips("验证码错误，请重新输入！", delay);
                    $("#imgCheckCommentCode").attr("src", "/code/" + parseInt(10000 * Math.random()) + "_conmment.html");
                } else if (data.status == 1001) {
                    art.dialog.tips("很抱歉，评论失败！", delay);
                } else {
                    art.dialog.tips("恭喜您，评论成功！", delay);
                    $('#orderCommentForm')[0].reset();//表单重置
                    //跳转页面
                    window.setTimeout(function () {
                        window.location.href = "/mobi/cn/member/order/0/1.html";
                    }, delay * 1000);

                    $("#imgCheckCommentCode").attr("src", "/code/" + parseInt(10000 * Math.random()) + "_conmment.html");
                }
            }
        });
        return false;
    });
});

//////////////////////////////找回密码//////////////////////////////////////
$(function () {
    $("#findPasswordForm").submit(function () {
        var delay = 1.5;
        ///用户名
        if ($("#txtUserName").val() == "") {
            $("#txtUserName").focus();
            art.dialog.tips("请输入用户名！", delay);
            return false;
        }
        ///验证码
        if ($("#txtCode").val() == "") {
            $("#txtCode").focus();
            art.dialog.tips("请输入验证码！", delay);
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/cn/find_password.ashx",
            data: "userName=" + encodeURI($("#txtUserName").val()) + "&code=" + encodeURI($("#txtCode").val()),
            success: function (obj) {
                if (obj.status == 1001) {
                    $("#txtCode").select();
                    art.dialog.tips("验证码错误！", delay);
                    $("#imgCheckCode").attr("src", "/code/" + parseInt(10000 * Math.random()) + "_findPwd.html");
                } else if (obj.status == 1002) {
                    $("#txtUserName").select();
                    art.dialog.tips("用户名不存在！", delay);
                } else {
                    $("#findPasswordForm").hide();
                    if (obj.data == "email") {
                        $("#divFindPassword").html('<div class="widthMAX MarginTop10"><div class="memberWarning">发送激活邮件至您的邮箱找回密码？</div></div><div class="widthMAX"><div class="LoginBtn_content"><input type="submit" id="btnSendMail" value="确认发送" class="LoginBtn"></div></div>').show();
                    } else {
                        $("#divFindPassword").show();
                        $("#span-question").html(obj.data);
                        $("#securityIssueForm").submit(function () {
                            if ($("#txtAnswer").val() == "") {
                                $("#txtAnswer").focus();
                                art.dialog.tips("请回答安全问题！", delay);
                                return false;
                            } else {
                                $.ajax({
                                    type: "POST",
                                    url: "/ashx/cn/find_password.ashx",
                                    data: "userName=" + encodeURI($("#txtUserName").val()) + "&answer=" + encodeURI($("#txtAnswer").val()),
                                    success: function (obj) {
                                        if (obj.status == 1001) {
                                            $("#txtAnswer").select();
                                            art.dialog.tips("很抱歉，回答错误！", delay);
                                            return false;
                                        } else {
                                            location.href = "/mobi/cn/password/update.html?data=" + obj.data;
                                            return false;
                                        }
                                    }
                                });
                            }
                            return false;
                        });
                    }
                    $("#btnSendMail").click(function () {
                        art.dialog.tips("正在发送邮件，请稍候  <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/find_password.ashx",
                            global: false,
                            async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                            data: "userName=" + encodeURI($("#txtUserName").val()),
                            success: function (obj) {
                                if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                                if (obj.status == 1000) {
                                    art.dialog.tips("发送邮件成功，请查收邮件！", delay);
                                    setTimeout("window.location.href='/mobi/cn/login.html'", delay * 1000)

                                } else {
                                    art.dialog.tips("发送邮件失败，请刷新后重试！", delay);
                                }
                            }
                        });
                        return false;
                    });
                }
            }
        });
        return false;
    });

});

//////////////////////////////重设密码:搬到user.js//////////////////////////////////////
$(function () {
    $("#btnUpdatePassword").click(function () {
        var delay = 1.5;
        ///用户密码
        if ($("#txtNewPwd").val() == "") {
            $("#txtNewPwd").focus();
            art.dialog.tips("请输入新密码！", delay);
            return false;
        } else {
            if ($("#txtNewPwd").val().length < 6 || $("#txtNewPwd").val().length > 16) {
                $("#txtNewPwd").select();
                art.dialog.tips("密码长度不符，必须在6-16个字符之间！", delay);
                return false;
            }
        }
        ///确认密码
        if ($("#txtAgainPwd").val() == "") {
            $("#txtAgainPwd").focus();
            art.dialog.tips("请输入确认密码！", delay);
            return false;
        } else {
            if ($("#txtAgainPwd").val() != $("#txtNewPwd").val()) {
                $("#txtAgainPwd").select();
                art.dialog.tips("确认密码不一致！", delay);
                return false;
            }
        }
        $.ajax({
            type: "POST",
            url: "/ashx/cn/update_password.ashx",
            data: "data=" + getUrlParam('data') + "&pwd=" + encodeURI($("#txtNewPwd").val()),
            success: function (obj) {
                if (obj.status == 1000) {
                    art.dialog.tips("密码重设成功！", 1.5);
                    setTimeout("window.location.href='/mobi/cn/login.html'", delay * 1000);
                } else {
                    art.dialog.tips("密码更新失败，请刷新后重试！", delay);
                }
            }
        });
        return false;
    });
});

//////////////////////////////重设分佣提现密码//////////////////////////////////////
$(function () {
    var delay = 1.5;
    $("#updateCommissionPasswordForm").submit(function () {
        ///用户密码
        if ($("#txtNewPwd").val() == "") {
            $("#txtNewPwd").focus();
            art.dialog.tips("请输入新密码！", delay);
            return false;
        } else {
            if ($("#txtNewPwd").val().length < 6 || $("#txtNewPwd").val().length > 16) {
                $("#txtNewPwd").select();
                art.dialog.tips("密码长度不符，必须在6-16个字符之间！", delay);
                return false;
            }
        }
        ///确认密码
        if ($("#txtAgainPwd").val() == "") {
            $("#txtAgainPwd").focus();
            art.dialog.tips("请输入确认密码！", delay);
            return false;
        } else {
            if ($("#txtAgainPwd").val() != $("#txtNewPwd").val()) {
                $("#txtAgainPwd").select();
                art.dialog.tips("确认密码不一致！", delay);
                return false;
            }
        }
        $.ajax({
            type: "POST",
            url: "/ashx/user_CommissionPassword.ashx",
            //data:"data="+location.search.split('=')[1]+"&pwd="+escape($("#txtNewPwd").val()),
            data: "data=" + getUrlParam("data") + "&t=2&newPwd=" + encodeURI($("#txtNewPwd").val()),
            success: function (obj) {
                if (obj != null && obj != undefined && obj.status == 1000) {
                    art.dialog.tips("密码重设成功！", delay);
                    setTimeout("window.location.href='/mobi/cn/member/commission/index.html'", delay);
                } else {
                    art.dialog.tips("密码更新失败，请刷新后重试！", delay);
                }
            }
        });
        return false;
    });
});

//////////////////////////////积分兑换详细//////////////////////////////////////
function pointsFor(id, inventory) {
    var delay = 1.5;
    if (parseInt(inventory) <= 0) {
        art.dialog.tips("很抱歉，积分商品库存不足！", delay);
        return false;
    }
    $.ajax({
        type: "POST",
        url: "/ashx/cn/points_for.ashx",
        data: "gid=" + id + "&amount" + $("txtCount").val(),
        success: function (obj) {
            if (obj.status == 1001)
                window.location = "/mobi/cn/login.html";//没有登录
            else if (obj.status == 1000) {
                confirmMessage('您确定要兑换此物品吗？', '/mobi/cn/integral/order/submit/' + id + '_' + $("#txtCount").val().trim() + '.html');
            } else {
                art.dialog.tips("积分商品库存不足！", delay);
            }
        }
    });
}

$(function () {
    $("#integralOrderSumitForm").submit(function () {
        var delay = 1.5;
        if ($("input[name='userAddress']:checked").val() == null) {
            $("input[name='userAddress']")[0].focus();
            art.dialog.tips("请选择或填写收货地址！", delay);
            return false;
        }
        if ($("input[name='userAddress']:checked").val() == "0") {
            if ($("#txtConsignee").val() == "") {
                $("#txtConsignee").focus();
                art.dialog.tips("请输入收货人姓名！", delay);
                return false;
            }
            if ($("#selectProvinces").val() == "0") {
                $("#selectProvinces").focus();
                art.dialog.tips("请选择省份/直辖市！", delay);
                return false;
            }
            if ($("#selectCity") != null) {
                if ($("#selectCity").val() == "0") {
                    $("#selectCity").focus();
                    art.dialog.tips("请选择市(县/区)！", delay);
                    return false;
                }
            }
            if ($("#txtAddress").val() == "") {
                $("#txtAddress").focus();
                art.dialog.tips("请输入详细地址！", delay);
                return false;
            }
            if ($("#txtZipcode").val() !== "" && !formatCheck($("#txtZipcode").val(), "zipCode")) {
                $("#txtZipcode").focus();
                art.dialog.tips("邮政编码格式不符！", delay);
                return false;
            }
            if ($("#txtTelephone").val() === "" && $("#txtMobile").val() === "") {
                $("#txtTelephone").focus();
                art.dialog.tips("联系电话和手机号码至少填写一项！", delay);
                return false;
            }
            if ($("#txtTelephone").val() !== "" && !formatCheck($("#txtTelephone").val(), "telephone")) {
                $("#txtTelephone").focus();
                art.dialog.tips("电话号码格式不符！如：0754-88888888", delay);
                return false;
            }
            if ($("#txtMobile").val() !== "" && !formatCheck($("#txtMobile").val(), "mobile")) {
                $("#txtMobile").focus();
                art.dialog.tips("手机号码格式不符！", delay);
                return false;
            }
        }
        if ($("#txtaPostscript").val().length > 200) {
            $("#txtaPostscript").select();
            art.dialog.tips("附言不能超过200个字符！", delay);
            return false;
        }
        var da = "";
        if ($("input[name='userAddress']").size() > 0) {
            da = "&userAddress=" + $("input[name='userAddress']:checked").val();
        } else {
            art.dialog.tips("请填写收货地址！", delay);
            setTimeout("window.location.href='/mobi/cn/member/shipping/address_0.html'", delay * 1000);
        }
        if (parseInt($("#hdGoodsSum").val()) == 0) {
            art.dialog.tips("购物车没有加入任何商品！", delay);
            var lurl = "window.location.href='/mobi/cn/index.html'";
            setTimeout(lurl, delay * 1000);
            return false;
        }
        art.dialog.tips("订单提交中，请耐心等候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
        $.ajax({
            type: "POST",
            url: "/ashx/cn/points_for_info.ashx",
            data: da + "&postscript=" + encodeURI($("#txtaPostscript").val()) + "&id=" + $("#hdPfId").val() + "&amount=" + $("#hdAmount").val(),
            success: function (obj) {
                if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                if (obj.status == 1000)
                    window.location = "/mobi/cn/integral/order/result/" + obj.data.orderNo + ".html";
                else if (obj.status == 1001)
                    art.dialog.tips("很抱歉，积分商品库存不足！", delay);
                else if (obj.status == 1002) {
                    art.dialog.tips("很抱歉，订单提交失败！", delay);
                    setTimeout("window.location.reload();", delay * 1000);
                }
            }
        });
        return false;
    });
});

$(function () {
    $("#btnConvertibilityOrderPay").click(function () {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/ashx/cn/points_for_order_pay.ashx",
            data: "orderNo=" + $("#hdfOrderNo").val(),
            success: function (obj) {
                if (obj.status == 1001) {
                    art.dialog.tips("您的消费积分不足以兑换订单积分！", delay);
                } else if (obj.status == 1002) {
                    art.dialog.tips("你的订单已经兑换成功，请不要重复兑换！", delay);
                    setTimeout("window.location.reload()", delay * 1000);
                } else if (obj.status == 1000) {
                    art.dialog.tips("恭喜您，兑换成功！", delay);
                    $.ajax({
                        type: "POST",
                        url: "/ashx/cn/async_send_message.ashx",
                        data: "orderNo=" + $("#hdfOrderNo").val() + "&sendType=point",
                        global: false,
                        async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                        success: function () {
                            //location.href="/cn/center/member/bulk/order/"+$("#hdfOrderNo").val()+"_info.html";
                            //setTimeout("window.location.reload();", delay * 1000);
                        }
                    });
                    setTimeout("window.location.reload();", delay * 1000);
                }
            }
        });
        return false;
    });

    $("#integralGoodsOrderForm").submit(function () {
        var orderno = $("#txtOrderNo").val();
        if (orderno != '')
            window.location = "/mobi/cn/member/convertibility/order" + $("input[name='status']").val() + "/1/" + orderno + ".html";
        else
            window.location = "/mobi/cn/member/convertibility/order/" + $("input[name='status']").val() + "/1.html";
        return false;
    });

    $("#aConvertibilityOrderPay").click(function () {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/ashx/cn/points_for_order_pay.ashx",
            data: "orderNo=" + $("#hdfOrderNo").val(),
            success: function (data) {
                if (data.status == 1001) {
                    art.dialog.tips("您的消费积分不足以兑换订单积分！", delay);
                } else if (data.status == 1002) {
                    art.dialog.tips("你的订单已经兑换成功，请不要重复兑换！", delay);
                    setTimeout("window.location.reload()", delay * 1000);
                } else if (data.status == 1000) {
                    art.dialog.tips("恭喜您，兑换成功！", delay);
                    setTimeout("window.location.reload()", delay * 1000);
                }
            }
        });
        return false;
    });
});

$(function () {
    //  微信会员卡
    $("#formMemberCard").submit(function () {
        var delay = 1.5;
        if ($("input[name='bind']:checked").val() == "1") {
            if ($("#txtUserName").val() == "") {
                $("#txtUserName").select();
                art.dialog.tips("请输入用户名！", delay);
                return false;
            }
            if ($("#txtUserPwd").val() == "") {
                $("#txtUserPwd").select();
                art.dialog.tips("请输入用户密码！", delay);
                return false;
            }
            $.ajax({
                type: "POST",
                url: "/ashx/cn/member_card.ashx",
                data: "userName=" + encodeURI($("#txtUserName").val()) + "&userPwd=" + $("#txtUserPwd").val() + "&memberNumber=" + $("#hdMemberNumber").val() + "&parentId=" + $("#hdParentId").val(),
                success: function (data) {
                    if (data.status == 1001) {
                        $("#txtUserName").select();
                        art.dialog.tips("用户名不存在！", delay);
                    } else if (data.status == 1002) {
                        $("#txtUserPwd").select();
                        art.dialog.tips("用户密码错误！", delay);
                    } else if (data.status == 1003) {
                        $("#txtUserName").select();
                        art.dialog.tips("该用户被停用！", delay);
                    } else if (data.status == 1004) {
                        $("#txtUserName").select();
                        art.dialog.tips("该用户还没有通过审核，请耐心等待或联系管理员！", delay);
                    } else if (data.status == 1000) {
                        art.dialog.tips("恭喜您，会员绑定成功！", delay);
                        setTimeout("location.href='/mobi/cn/member/';", delay * 1000);
                    } else if (data.status == 1005) {
                        $("#txtUserName").select();
                        art.dialog.tips("绑定失败,信息有误！", delay);
                    }
                }
            });
        } else if ($("input[name='bind']:checked").val() == "0") {
            ///用户名
            if ($("#txtRegisterUserName").val() == "") {
                $("#txtRegisterUserName").focus();
                art.dialog.tips("请输入用户名！", delay);
                return false;
            } else {
                if ($("#txtRegisterUserName").val().length < 4 || $("#txtRegisterUserName").val().length > 16) {
                    $("#txtRegisterUserName").select();
                    art.dialog.tips("用户名长度不符！", delay);
                    return false;
                }
            }
            ///用户密码
            if ($("#txtRegisterUserPwd").val() == "") {
                $("#txtRegisterUserPwd").focus();
                art.dialog.tips("请输入用户密码！", delay);
                return false;
            } else {
                if ($("#txtRegisterUserPwd").val().length < 6 || $("#txtRegisterUserPwd").val().length > 16) {
                    $("#txtRegisterUserPwd").select();
                    art.dialog.tips("密码长度不符，必须在6-16个字符之间！", delay);
                    return false;
                }
            }
            ///确认密码
            if ($("#txtRegisterAgainPwd").val() == "") {
                $("#txtRegisterAgainPwd").focus();
                art.dialog.tips("请输入确认密码！", delay);
                return false;
            } else {
                if ($("#txtRegisterAgainPwd").val() != $("#txtRegisterUserPwd").val()) {
                    $("#txtRegisterAgainPwd").select();
                    art.dialog.tips("确认密码不一致！", delay);
                    return false;
                }
            }
            ///Email
            if (!formatCheck($("#txtEmail").val(), "email")) {
                $("#txtEmail").focus();
                art.dialog.tips("电子邮箱格式不符！", delay);
                return false;
            }
            $.ajax({
                type: "POST",
                url: "/ashx/cn/member_card.ashx",
                data: "userName=" + encodeURI($("#txtRegisterUserName").val()) + "&userPwd=" + $("#txtRegisterUserPwd").val() + "&email=" + encodeURI($("#txtEmail").val()) + "&memberNumber=" + $("#hdMemberNumber").val() + "&parentId=" + $("#hdParentId").val(),
                success: function (data) {
                    if (data.status == 1001) {
                        art.dialog.tips("很抱歉，注册失败！", delay);
                    } else if (data.status == 1002) {
                        $("#txtRegisterUserName").select();
                        art.dialog.tips("用户名已存在！", delay);
                    } else if (data.status == 1003) {
                        $("#txtEmail").select();
                        art.dialog.tips("电子邮箱已注册！", delay);
                    } else {
                        if (data.status == 1000) {
                            art.dialog.tips("恭喜您，注册成功，会员绑定成功！", delay);
                            $.ajax({
                                type: "POST",
                                url: "/ashx/cn/async_register_message.ashx",
                                data: "userName=" + encodeURI($("#txtRegisterUserName").val()),
                                global: false,
                                async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                                success: function () {
                                }
                            });
                            setTimeout("location.href='/mobi/cn/member/'", 3000);
                        } else if (data.status == 1004) {
                            art.dialog.tips("该用户还没有通过审核，请耐心等待或联系管理员！", delay);
                        }
                    }
                    return false;
                }
            });
        } else {
            $("input[name='bind']")[0].focus();
            art.dialog.tips("请选择是否绑定会员！", delay);
        }
        return false;
    });
    $("input[name='bind']").change(function () {
        if ($("input[name='bind']:checked").val() == "1") {
            $("#spOldUser").show();
            $("#spNewUser").hide();
        } else {
            $("#spOldUser").hide();
            $("#spNewUser").show();
        }
    });
});

//////////////////////////////团购订单详细//////////////////////////////////////
$(function () {
    $("#btnBulkOrderPay").click(function () {
        var paybank = $("input[type='hidden'][name='pay_bank']").val()
            , $self = $(this)
            , delay = 5
        ;
        if ($("input[name='pay_bank']:checked").length == 0) {
            $("input[name='pay_bank']")[0].focus();
            art.dialog.tips("请选择支付方式！", delay);
            return false;
        }
        $(this).addClass("disabled").attr("data-text", $self.html()).html("正在支付");
        if ($("input[name='pay_bank']:checked").val() == "预存款") {
            art.dialog.tips("正在提交数据，请稍候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
            $.ajax({
                type: "POST",
                url: "/ashx/cn/group_buy_order_pay.ashx",
                data: "orderNo=" + $("#hdfOrderNo").val() + "&payment=" + encodeURI("预存款"),
                success: function (msg) {
                    if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                    if (msg.status == 1001) {
                        art.dialog.tips("您的余额不足以支付订单金额！<a target=\"_blank\" href=\"/mobi/cn/member/pay/online.html\" style=\"color:Red; font-weight:bold;\">立即充值</a>", delay);
                    } else if (msg.status == 1002) {
                        art.dialog.tips("你的订单已经支付，请不要重复支付！", delay);
                        setTimeout("window.location.reload()", delay * 1000);
                    } else if (msg.status == 1000) {
                        art.dialog.tips("恭喜您，支付成功！", delay);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/async_send_message.ashx",
                            data: "orderNo=" + $("#hdfOrderNo").val() + "&sendType=2",
                            global: false,
                            async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                            success: function () {
                                //window.location.reload();
                            }
                        });
                        setTimeout(function () {
                            window.location.href = "/mobi/cn/member/bulk/order/" + $("#hdfOrderNo").val() + "_info.html";
                        }, delay * 1000);
                    }
                }
            });
        } else {
            payment($("input[name='pay_bank']:checked").val(), $("#hdfOrderNo").val(), {
                hdfWxpay: $("#hdfWxpay").val(),
                hdfMicroMessenger: $("#hdfMicroMessenger").val()
            }, null, $("orderTotal").val(), function (error) {
                $self.removeClass("disabled").html($self.attr("data-text"));
            });
        }
        return false;
    });
});
///////////////////团购订单提交//////////////////////
$(function () {
    $("#bulkOrderSumitForm").submit(function () {
        var userInfo = ""
            , da = ""
            , delay = 5
        ;
        if ($("input[name='userAddress']:checked").val() == null) {
            $("input[name='userAddress']").eq(0).focus();
            art.dialog.tips("请选择或填写收货地址！", delay);
            return false;
        }
        if ($("input[name='userAddress']:checked").val() == "0") {
            if ($("#txtConsignee").val() == "") {
                $("#txtConsignee").focus();
                art.dialog.tips("请输入收货人姓名！", delay);
                return false;
            }
            if ($("#selectProvinces").val() == "0") {
                $("#selectProvinces").focus();
                art.dialog.tips("请选择省份/直辖市！", delay);
                return false;
            }
            if ($("#selectCity") != null) {
                if ($("#selectCity").val() == "0") {
                    $("#selectCity").focus();
                    art.dialog.tips("请选择市(县/区)！", delay);
                    return false;
                }
            }
            if ($("#txtAddress").val() == "") {
                $("#txtAddress").focus();
                art.dialog.tips("请输入详细地址！", delay);
                return false;
            }
            if ($("#txtZipcode").val() !== "" && !formatCheck($("#txtZipcode").val(), "zipCode")) {
                $("#txtZipcode").focus();
                art.dialog.tips("邮政编码格式不符！", delay);
                return false;
            }
            if ($("#txtTelephone").val() === "" && $("#txtMobile").val() === "") {
                $("#txtTelephone").focus();
                art.dialog.tips("联系电话和手机号码至少填写一项！", delay);
                return false;
            }
            if ($("#txtTelephone").val() !== "" && !formatCheck($("#txtTelephone").val(), "telephone")) {
                $("#txtTelephone").focus();
                art.dialog.tips("电话号码格式不符！如：0754-88888888", delay);
                return false;
            }
            if ($("#txtMobile").val() !== "" && !formatCheck($("#txtMobile").val(), "mobile")) {
                $("#txtMobile").focus();
                art.dialog.tips("手机号码格式不符！", delay);
                return false;
            }
        }

        if ($("input[name='deliveryWay']:checked").val() == null) {
            $("input[name='deliveryWay']")[0].focus();
            art.dialog.tips("请选择配送方式！", delay);
            return false;
        }
        //        if ($("input[name='pay_bank']:checked").val() == null) {
        //            $("input[name='pay_bank']")[0].focus();
        //            art.dialog.tips("请选择支付方式！", delay);
        //            return false;
        //        }
        if ($("#txtaPostscript").val().length > 200) {
            $("#txtaPostscript").select();
            art.dialog.tips("附言不能超过200个字符！", delay);
            return false;
        }
        if ($("input[name='userAddress']:checked").val() == "0") {
            if ($("#selectCity option:selected").text() != "")
                da = "&userAddress=0&consignee=" + encodeURI($("#txtConsignee").val()) + "&provinces=" + encodeURI($("#selectProvinces option:selected").text()) + "&city=" + encodeURI($("#selectCity option:selected").text()) + "&address=" + encodeURI($("#txtAddress").val()) + "&zipCode=" + $("#txtZipcode").val() + "&telephone=" + $("#txtTelephone").val() + "&mobile=" + $("#txtMobile").val() + "&isSave=" + $("input[name='isSave']:checked").val();
            else
                da = "&userAddress=0&consignee=" + encodeURI($("#txtConsignee").val()) + "&provinces=" + encodeURI($("#selectProvinces option:selected").text()) + "&city=&address=" + encodeURI($("#txtAddress").val()) + "&zipCode=" + $("#txtZipcode").val() + "&telephone=" + $("#txtTelephone").val() + "&mobile=" + $("#txtMobile").val() + "&isSave=" + $("input[name='isSave']:checked").val();
        } else {
            da = "&userAddress=" + $("input[name='userAddress']:checked").val();
        }
        if (parseInt($("#hdGoodsSum").val()) == 0) {
            art.dialog.tips("购物车没有加入任何商品！", delay);
            var lurl = "window.location.href='/mobi/cn/index.html'";
            setTimeout(lurl, delay * 1000);
            return false;
        }
        //团购订单提交
        art.dialog.tips("订单提交中，请耐心等候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
        $.ajax({
            type: "POST",
            url: "/ashx/cn/group_buy_order_info.ashx",
            data: "deliveryWay=" + $("input[name='deliveryWay']:checked").val() + da + "&postscript=" + encodeURI($("#txtaPostscript").val()) + "&goodsSum=" + $("#hdGoodsSum").val() + ($("#hdAnonymous").length > 0 ? "&Anonymous=1" : "") + "&isInvoice=" + $("#cbInvoice").is(':checked'),
            success: function (obj) {
                if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                if (obj.status == -1001) {
                    art.dialog.tips("很抱歉，您购买的商品库存不足,请返回购物车修改此商品数量！<br />" + obj.data.goodsName, delay);
                    return false;
                } else if (obj.status == -1002) {
                    art.dialog.tips("很抱歉，您购买的商品已经下架,请返回购物车删除此商品！<br />" + obj.data.goodsName, delay);
                    return false;
                } else if (obj.status == -1003) {
                    art.dialog.tips("很抱歉，您购买的商品会员价已更改,请刷新后再提交订单！<br />" + obj.data.goodsName, delay);
                    setTimeout("window.location.href='/mobi/cn/bulk/order/submit.html'", delay * 1000);
                    return false;
                } else if (obj.status == 1004) {
                    art.dialog.tips("团购购物车目前没有加入任何商品！", 1.5);
                    setTimeout("window.location.href='/mobi/cn/bulk/order/submit.html'", delay * 1000);
                    return false;
                } else if (obj.status == -1004) {
                    art.dialog.tips("缺少收货人信息！", 1.5);
                    setTimeout("window.location.href='/mobi/cn/bulk/order/submit.html'", delay * 1000);
                    return false;
                } else {
                    if (obj.status == 1000) {
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/async_send_message.ashx",
                            data: "orderNo=" + obj.data.orderNo + "&sendType=1",
                            global: false,
                            async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                            success: function () {
                                location.href = "/mobi/cn/bulk/order/result/" + obj.data.orderNo + ".html";
                            }
                        });
                        setTimeout("window.location.href='/mobi/cn/bulk/order/result/" + obj.data.orderNo + ".html';", delay * 1000);
                    } else {
                        art.dialog.tips("很抱歉，订单提交失败！", delay);
                        var lurl = "window.location.href='/mobi/cn/bulk/order/submit.html'";
                        setTimeout(lurl, delay * 1000);
                        return false;
                    }
                }
            }
        });
        return false;
    });
});

//////////////////////////////秒杀订单详细及秒杀订单支付//////////////////////////////////////

$(function () {
    $("#btnSeckillOrderPay").click(function () {
        var $self = $(this)
            , delay = 5
            , waitDelay = 30
            , orderNo = $("#hdfOrderNo").val()  //  订单号
            , orderTotal = $("#orderTotal").val()  //  订单金额
            , payBank = $("input[name=pay_bank]:checked")  //  支付方式对象
            , payBankName = payBank.attr("data-name")  //  支付方式名称
            , payPwdData = ""  //  支付方式名称
            , payUrl = getApiByOrderNo(orderNo)    //  支付路径
            , pay_bank
            , disabledText = "正在支付"
        ;

        if ($(this).hasClass("disabled")) {
            art.dialog.tips(disabledText, delay);
            return false;
        }

        if (payBank.length == 0) {
            $("input[name=pay_bank]").focus();
            art.dialog.tips("请选择支付方式！", delay);
            return false;
        }

        pay_bank = payBank.val();
        $(this).addClass("disabled").attr("data-text", $self.html()).html(disabledText);
        var integralParaStr = ($("#cbIntegral").length == 0) ? "" : "&cbInteralCheck=" + ($("#cbIntegral").is(":checked") ? 1 : 0) + "&orderIntegralPayAmount=" + $("#orderIntegralPayAmount").val() + "&userIntegralForOrder=" + $("#userIntegralForOrder").val();
        if ($("#cbPreDeposit").is(':checked') || ($("#cbIntegral").length == 1 && $("#cbIntegral").is(":checked") && payBank.val() != "预存款")) {
            $.ajax({
                type: "POST",
                url: payUrl,
                data: "orderNo=" + orderNo + "&pd=" + ($("#cbPreDeposit").is(":checked") ? 1 : 0) + integralParaStr + "&prepare=1",
                success: function (obj) {
                    var orderTotal = parseFloat(obj.data);
                    if (orderTotal >= 0) {
                        $.ajax({
                            type: "POST",
                            url: payUrl,
                            data: "orderNo=" + orderNo + "&payment=" + encodeURI("预存款") + integralParaStr,
                            success: function (data) {
                                $self.removeClass("disabled").html($self.attr("data-text"));
                                orderPaySuccessCB(data, orderNo, delay);
                            }
                        });
                    } else {
                        orderTotal = orderTotal * (-1);
                        payment(pay_bank, orderNo, {
                            hdfWxpay: $("#hdfWxpay").val(),
                            hdfMicroMessenger: $("#hdfMicroMessenger").val()
                        }, null, orderTotal, function (error) {
                            $self.removeClass("disabled").html($self.attr("data-text"));
                        });
                    }
                }
            });
            return false;
        } else {
            //第三方支付
            $.ajax({
                type: "POST",
                url: "/ashx/cn/order_status.ashx",
                data: "orderNo=" + $("#hdfOrderNo").val(),
                success: function (obj) {
                    switch (obj.status) {
                        case 1000:
                            //订单正常,进入支付流程
                            payment(payBank.val(), orderNo, {
                                hdfMicroMessenger: $("#hdfMicroMessenger").val(),
                                hdfWxpay: $("#hdfWxpay").val()
                            }, null, null, function (error) {
                                $self.removeClass("disabled").html($self.attr("data-text"));
                            });
                            break;
                        case 1001:
                            art.dialog.tips("支付超时,订单已取消！", 1.5);
                            setTimeout(function () {
                                window.location.reload();
                            }, 2000);
                            break;
                        case 1003:
                            art.dialog.tips("订单已支付,请不要重复支付！", 1.5);
                            setTimeout(function () {
                                window.location.reload();
                            }, 2000);
                            break;
                    }
                }
            });
        }
        if (payBank.val() == "货到付款") {
            art.dialog.tips("正在提交数据，请稍候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
            $.ajax({
                type: "POST",
                url: payUrl,
                data: "orderNo=" + orderNo + "&payment=" + encodeURI("货到付款"),
                success: function (msg) {
                    if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                    if (msg.status == 1000) {
                        art.dialog.tips("恭喜您，订单处理成功！", 3);
                        setTimeout(function () {
                            location.href = "/mobi/cn/member/seckill/order/" + orderNo + ".html";
                        }, delay * 1000);
                    } else if (msg.status == 1006) {
                        art.dialog.tips("操作超时,订单已取消!", 1.5);
                        setTimeout("window.location.reload()", 2000);
                    }
                }
            });
        }
        if (payBank.val() == "预存款") {
            art.dialog.tips("正在提交数据，请稍候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
            $.ajax({
                type: "POST",
                url: payUrl,
                data: "orderNo=" + orderNo + "&payment=" + escape("预存款") + integralParaStr + payPwdData,
                success: function (msg) {
                    if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                    $self.removeClass("disabled").html($self.attr("data-text"));
                    orderPaySuccessCB(msg, orderNo, delay);
                }
            });
        }
        return false;
    });
});
//=====app里点击第三方登录====
$(function () {
    $(".div-other a").click(function () {
        var t = $(this).attr("data-type");
        if (window.navigator.userAgent.toLowerCase().match("micronetloginapp") && t != "sina") {
            micronetlogin.onClickLogin(t);
            return false;
        }
    });
});

//========搜索==========
$(function () {
    $("#kewSearchForm").submit(function () {
        if ($("#txtKeyword").val() != "")
            location.href = "/mobi/cn/goods_list.html?kw=" + encodeURIComponent($("#txtKeyword").val().trim());
        return false;
    });
    $("#imgSearch").click(function () {
        if ($("#txtKeyword").val() != "")
            location.href = "/mobi/cn/goods_list.html?kw=" + encodeURIComponent($("#txtKeyword").val().trim());
        return false;
    });
});

///==============取消订单===================

$(function () {
    var delay = 1.5;
    $('.OrderListCancel').click(function () {
        var orderNo = $(this).data('orderno');
        art.dialog.confirm("您确定要取消订单吗？", function () {
            if (orderNo != '') {
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/order_cancel.ashx",
                    data: "orderNo=" + orderNo,
                    success: function (data) {
                        if (data.status == 1001) art.dialog.tips("取消订单失败！", delay);
                        else if (data.status == 1002) {
                            art.dialog.tips("订单已经取消，无需重复操作！", delay);
                            setTimeout(function () {
                                location.reload();
                            }, delay * 1000);
                        } else {
                            art.dialog.tips("取消订单成功！", delay)
                            setTimeout(function () {
                                location.reload();
                            }, delay * 1000);
                        }
                    }
                });
                return true;
            }
        }, function () {
            art.dialog.tips('你取消了操作');
        });
    });
});

//==============删除发件箱==================
$(function () {
    var delay = 1.5;
    $(".delOutbox").click(function () {
        var id = $(this).attr("data-id");
        art.dialog.confirm("您确定要删除吗？", function () {
            if (id != '') {
                $.ajax({
                    type: "POST",
                    url: "/outbox/delete",
                    data: "id=" + id,
                    success: function (data) {
                        if (data.status == 1001) art.dialog.tips("删除失败！", delay);
                        else if (data.status == 1000) {
                            art.dialog.tips("删除成功！", delay);
                            setTimeout(function () {
                                window.location.reload();
                            }, delay * 1000);
                        } else {
                            art.dialog.tips(data.msg, delay);
                        }
                    }
                });
                return true;
            }
        }, function () {
            art.dialog.tips('你取消了操作');
        });
    });
});
//==============设置默认收货地址==============
$(function () {
    var delay = 1.5;
    $(".setShipping").click(function () {
        var id = $(this).attr("data-id");
        if (id != '') {
            $.ajax({
                type: "POST",
                url: "/ashx/cn/shopping_address.ashx",
                data: "type=setDefault&did=" + id,
                success: function (data) {
                    if (data.status == 1001) art.dialog.tips("设置默认收货地址失败！", delay);
                    else if (data.status == 1000) {
                        art.dialog.tips("设置默认收货地址成功！", delay);
                        setTimeout(function () {
                            window.location.reload();
                        }, delay * 1000);
                    } else {
                        art.dialog.tips(data.msg, delay);
                    }
                }
            });
            return true;
        }
    });
});

//==============删除收货地址=================

function deleteAddress(obj) {
    var id = obj
        , delay = 5
    ;
    if (id != '') {
        $.ajax({
            type: "POST",
            url: "/ashx/cn/shopping_address.ashx",
            data: "type=del&did=" + id,
            success: function (data) {
                if (data.status == 1001) art.dialog.tips("删除收货地址失败！", delay);
                else if (data.status == 1000) {
                    art.dialog.tips("删除收货地址成功！", delay);
                    setTimeout(function () {
                        window.location.reload();
                    }, delay * 1000);
                } else {
                    art.dialog.tips(data.msg, delay);
                }
            }
        });
        return true;
    }
}

function checkDeliveryWay(freight) {
    freight = freight !== undefined ? freight : $("input[name='deliveryWay']:checked").attr("data-yunfei");
    var cpv = 0.00;
    if ($("#hdCouponParValue").val() != null && $("#hdCouponParValue").val() != undefined)
        cpv = parseFloat($("#hdCouponParValue").val()).toFixed(2);
    if ($("#hdFromFreight").val() == "false") {
        $("#sSumFreight").html(changePrice2money("" + parseFloat(freight).toFixed(2) + ""));
        $("#sSumGoods").html(changePrice2money("" + (parseFloat($("#hdGoodsTotal").val()) + parseFloat(freight)) + ""));
        $("#sSumTotal").html(changePrice2money("" + (parseFloat($("#hdGoodsSum").val()) + parseFloat(freight)) - cpv + ""));
    } else if ($("#hdFromFreight").val() == "true") {
        $("#sSumFreight").html(changePrice2money("" + parseFloat(0).toFixed(2) + ""));
        $("#sSumGoods").html(changePrice2money("" + (parseFloat($("#hdGoodsTotal").val()) + parseFloat(0)) + ""));
        $("#sSumTotal").html(changePrice2money("" + (parseFloat($("#hdGoodsSum").val()) + parseFloat(0) - cpv) + ""));
    }
}

//===============订单提交================
$(function () {
    if ($("input[name='deliveryWay']").length > 0) {
        $("input[name='deliveryWay']").eq(0).attr("checked", "checked");
        checkDeliveryWay();
    }
    //改变收货地址
    $("input[name='userAddress']").click(function () {
        var check = $(this).prop("checked"), poststr;
        var t = $("input[name='orderType']").val();
        var url = "/ashx/cn/order_info.ashx";
        if (t == "bulk") {
            url = "/ashx/cn/group_buy_order_info.ashx";
        } else if (t == "buygang") {
            url = "/ashx/cn/buy_gang_order_info.ashx";
        } else if (t == "spell") {
            url = "/ashx/cn/spell_order_info.ashx";
        }
        var data = "";
        if (url == "/ashx/cn/order_info.ashx") {
            data = '&geid=' + $("#hdGoodsGeid").val() + "&buyNowAmount=" + $("#hdBuyNowAmount").val();
        } else if (url == "/ashx/cn/spell_order_info.ashx") {
            data = '&geid=' + $("#hdGoodsGeid").val();
        }
        if (check && $(this).val() != "0" && t != null) {
            $("#addUserAddress").hide();
            //查询信息
            $.ajax({
                type: "POST",
                url: url,
                data: "region=" + encodeURI($(this).attr("data-region")) + data,
                dataType: "json",
                success: function (obj) {
                    //多个
                    $("#delivery-div").children().remove('dd');
                    for (var i = 0; i < obj.length; i++) {
                        var o = obj[i];
                        $("#delivery-div").append('<dd><div class="floatl">' + o.deliveryWay + ' +' + o.yunfei + '</div><div class="floatr"><input type="radio" name="deliveryWay" data-yunfei="' + o.yunfei + '" value="' + o.deliveryWay + ',' + o.yunfei + '" /></div></dd>');
                    }
                    $("input[name='deliveryWay']").eq(0).attr("checked", "checked");
                    checkDeliveryWay();
                }
            });
        }
        if (check && $(this).val() == "0") {
            $("#addUserAddress").show();
            if ($("#selectProvinces option:selected").val() != "0") {
                poststr = "region=" + encodeURI($("#selectProvinces option:selected").text());
                //查询信息
                $.ajax({
                    type: "POST",
                    url: url,
                    data: poststr,
                    dataType: "json",
                    success: function (obj) {
                        //多个
                        $("#delivery-div").children().remove();
                        for (var i = 0; i < obj.length; i++) {
                            var o = obj[i];
                            $("#delivery-div").append('<dd><div><div class="floatl">' + o.deliveryWay + ' +' + o.yunfei + '</div><div class="floatr"><input type="radio" name="deliveryWay" data-yunfei="' + o.yunfei + '" value="' + o.deliveryWay + ',' + o.yunfei + '" /></div></div></dd>');
                            //                            $("#delivery-div").append('<div class="Settlement_hang"><input type="radio" name="deliveryWay" data-yunfei="' + o.yunfei + '" value="' + o.deliveryWay + ',' + o.yunfei + '">' + o.deliveryWay + '<strong class="yunfei">+ ¥' + o.yunfei + '</strong></div>');
                        }
                    }
                });
            }
        }
    });

    ////点击运费修改费用/////
    $("#delivery-div").delegate("input[name='deliveryWay']", "click", function () {
        checkDeliveryWay();
    });
    if (location.href.indexOf("/mobi/cn/order/submit.html") != -1 || location.href.indexOf("mobi/cn/bulk/order/submit.html") != -1) {
        var date = new Date();
        var day = date.toLocaleDateString().replace(/\//g, "-");
        $("#txtStartDate").val(date.Format("yyyy-MM-dd"));
        $("#txtEndDate").val(date.Format("yyyy-MM-dd"));
        $("#pDeliveryDate").html(date.Format("yyyy-MM-dd") + '&nbsp;至&nbsp;' + date.Format("yyyy-MM-dd"));
        if ($("#txtStartDate").val() != "" && $("#txtEndDate").val() != "" && $("#txtStartDate").val() != undefined && $("#txtEndDate").val() != undefined) {
            var d1Arr = $("#txtStartDate").val().split('-');
            var d2Arr = $("#txtEndDate").val().split('-');
            var v1 = new Date(d1Arr[0], d1Arr[1], d1Arr[2]);
            var v2 = new Date(d2Arr[0], d2Arr[1], d2Arr[2]);
            if (v1 > v2) {
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/delivery_time.ashx",
                    dataType: "json",
                    data: "week=",
                    success: function (data) {
                        $("#DeliveryTime").children().remove();
                    }
                });
                return false;
            } else {
                getWeek();
            }
        }
    }
});


////伙拼订单提交
$(function () {
    var delay = 1.5;
    $("#buyGangOrderSumitForm").submit(function () {
        if (parseInt($("#hdGoodsSum").val()) == 0) {
            art.dialog.tips("购物车没有加入任何商品！", delay);
            var lurl = "window.location.href='/mobi/index.html'";
            setTimeout(lurl, delay * 1000);
            return false;
        }
        if ($("input[name='userAddress']:checked").val() == null) {
            $("input[name='userAddress']").eq(0).focus();
            art.dialog.tips("请选择或填写收货地址！", delay);
            return false;
        }
        if ($("input[name='userAddress']:checked").val() == "0") {
            if ($("#txtConsignee").val() == "") {
                $("#txtConsignee").focus();
                art.dialog.tips("请输入收货人姓名！", delay);
                return false;
            }
            if ($("#selectProvinces").val() == "0") {
                $("#selectProvinces").focus();
                art.dialog.tips("请选择省份/直辖市！", delay);
                return false;
            }
            if ($("#selectCity") != null) {
                if ($("#selectCity").val() == "0") {
                    $("#selectCity").focus();
                    art.dialog.tips("请选择市(县/区)！", delay);
                    return false;
                }
            }
            if ($("#txtAddress").val() == "") {
                $("#txtAddress").focus();
                art.dialog.tips("请输入详细地址！", delay);
                return false;
            }
            if (!formatCheck($("#txtZipcode").val(), "zipCode")) {
                $("#txtZipcode").focus();
                art.dialog.tips("请输入格式正确的邮政编码！", delay);
                return false;
            }
            if ($("#txtTelephone").val() === "" && $("#txtMobile").val() === "") {
                $("#txtTelephone").focus();
                art.dialog.tips("联系电话和手机号码至少填写一项！", delay);
                return false;
            }
            if ($("#txtTelephone").val() !== "" && !formatCheck($("#txtTelephone").val(), "telephone")) {
                $("#txtTelephone").focus();
                art.dialog.tips("电话号码格式不符！如：0754-88888888", delay);
                return false;
            }
            if ($("#txtMobile").val() !== "" && !formatCheck($("#txtMobile").val(), "mobile")) {
                $("#txtMobile").focus();
                art.dialog.tips("手机号码格式不符！", delay);
                return false;
            }
        }
        if ($("input[name='deliveryWay']:checked").val() == null) {
            $("input[name='deliveryWay']")[0].focus();
            art.dialog.tips("请选择配送方式！", delay);
            return false;
        }
        var deliveryDate = "";
        if ($("#txtStartDate").val() == "") {
            art.dialog.tips("请输入开始日期！", delay);
            $("#txtStartDate").select();
            return false;
        }
        if ($("#txtEndDate").val() == "") {
            art.dialog.tips("请输入结束日期！", delay);
            $("#txtEndDate").select();
            return false;
        }
        if ($("#txtStartDate").val() != "" && $("#txtEndDate").val() != "" && $("#txtStartDate").val() != undefined && $("#txtEndDate").val() != undefined) {
            var d1Arr = $("#txtStartDate").val().split('-');
            var d2Arr = $("#txtEndDate").val().split('-');
            var v1 = new Date(d1Arr[0], d1Arr[1], d1Arr[2]);
            var v2 = new Date(d2Arr[0], d2Arr[1], d2Arr[2]);
            var dateS = new Date(Date.parse(($("#txtStartDate").val() + " 23:59:59").replace(/-/g, '/')));
            var dateE = new Date(Date.parse(($("#txtEndDate").val() + " 23:59:59").replace(/-/g, '/')));
            var dateNow = new Date();
            if (dateS < dateNow && dateE < dateNow) {
                art.dialog.tips("送货日期已过期，请重新选择！", delay);
                $("#txtStartDate").select();
                return false;
            }
            if (v1 > v2) {
                $("#txtEndDate").select();
                art.dialog.tips("结束日期不能小于开始日期！", delay);
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/delivery_time.ashx",
                    data: "week=",
                    success: function (data) {
                        $("#DeliveryTime").children().remove();
                    }
                });
                return false;
            } else {
                if ($("#hdDeliveryWay").val() == "1") {
                    deliveryDate = $("#txtStartDate").val() + "至" + $("#txtEndDate").val();
                }
            }
        }
        var deliveryTime = "";
        var isDelivery = "";
        if ($("#hdDeliveryWay").val() == "1") {
            if ($("input[name='cbDeliveryTime']:checked").val() != null) {
                $("input[name='cbDeliveryTime']").each(function () {
                    if ($(this).prop("checked")) {
                        if (deliveryTime != "")
                            deliveryTime += "," + $(this).val();
                        else
                            deliveryTime += $(this).val();
                        var cbIsDelivery = "cbIsDelivery" + $(this).val();
                        if ($("#" + cbIsDelivery).prop("checked")) {
                            if (isDelivery != "")
                                isDelivery += ",1";
                            else
                                isDelivery += "1";
                        } else {
                            if (isDelivery != "")
                                isDelivery += ",0";
                            else
                                isDelivery += "0";
                        }
                    }
                });
            }
        }
        //        if ($("input[name='pay_bank']:checked").val() == null) {
        //            $("input[name='pay_bank']")[0].focus();
        //            art.dialog.tips("请选择支付方式！", delay);
        //            return false;
        //        }
        if ($("#txtaPostscript").val().length > 200) {
            $("#txtaPostscript").select();
            art.dialog.tips("附言不能超过200个字符！", delay);
            return false;
        }
        var da = "";
        if ($("input[name='userAddress']:checked").val() == "0") {
            if ($("#selectCity option:selected").text() != "")
                da = "&userAddress=0&consignee=" + encodeURI($("#txtConsignee").val()) + "&provinces=" + encodeURI($("#selectProvinces option:selected").text()) + "&city=" + encodeURI($("#selectCity option:selected").text()) + "&address=" + encodeURI($("#txtAddress").val()) + "&zipCode=" + $("#txtZipcode").val() + "&telephone=" + $("#txtTelephone").val() + "&mobile=" + $("#txtMobile").val() + "&isSave=" + $("input[name='isSave']:checked").val();
            else
                da = "&userAddress=0&consignee=" + encodeURI($("#txtConsignee").val()) + "&provinces=" + encodeURI($("#selectProvinces option:selected").text()) + "&city=&address=" + encodeURI($("#txtAddress").val()) + "&zipCode=" + $("#txtZipcode").val() + "&telephone=" + $("#txtTelephone").val() + "&mobile=" + $("#txtMobile").val() + "&isSave=" + $("input[name='isSave']:checked").val();
        } else {
            da = "&userAddress=" + $("input[name='userAddress']:checked").val();
        }
        art.dialog.tips("订单提交中，请耐心等候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
        $.ajax({
            type: "POST",
            url: "/ashx/cn/buy_gang_order_info.ashx",
            data: "deliveryWay=" + $("input[name='deliveryWay']:checked").val() + "&pay_bank=" + $("input[name='pay_bank']:checked").val() + da + "&postscript=" + encodeURI($("#txtaPostscript").val()) + "&goodsSum=" + $("#hdGoodsSum").val() + "&deliveryDate=" + deliveryDate + "&deliveryTime=" + deliveryTime + "&isDelivery=" + isDelivery + ($("#hdAnonymous").length > 0 ? "&Anonymous=1" : "") + "&isInvoice=" + $("#cbInvoice").is(':checked'),
            success: function (obj) {
                if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                if (obj.status == -1001) {
                    art.dialog.tips("很抱歉，您购买的伙拼商品库存不足,请返回购物车修改此商品数量！<br />" + obj.data.goodsName, delay);
                    return false;
                } else if (obj.status == -1002) {
                    art.dialog.tips("很抱歉，您购买的商品已经下架,请返回购物车删除此商品！<br />" + obj.data.goodsName, delay);
                    return false;
                } else if (obj.status == -1003) {
                    art.dialog.tips("很抱歉，您购买的伙拼商品伙拼价已更改,请刷新后再提交订单！<br />" + obj.data.goodsName, delay);
                    var lurl = "window.location.href='/mobi/cn/buy_gang/order/submit.html" + location.search + "'";
                    setTimeout(lurl, delay * 1000);
                    return false;
                } else {
                    if (obj.status == 1000) {
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/async_send_message.ashx",
                            data: "orderNo=" + obj.data.orderNo + "&sendType=1",
                            global: false,
                            async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                            success: function () {
                                location.href = "/mobi/cn/buy_gang/order/result/" + obj.data.orderNo + ".html";
                            }
                        });
                        setTimeout(function () {
                            location.href = "/mobi/cn/buy_gang/order/result/" + obj.data.orderNo + ".html";
                        }, delay * 1000);
                    } else {
                        art.dialog.tips("很抱歉，订单提交失败！", delay);
                        var lurl = "window.location.href='/mobi/cn/buy_gang/order/submit.html" + location.search + "'";
                        setTimeout(lurl, delay * 1000);
                        return false;
                    }
                }
            }
        });
        return false;
    });
});


//////////////////伙拼订单支付///////////////////////////
$(function () {
    var delay = 1.5;
    $("#btnBuyGangOrderPay").click(function () {
        var $self = $(this)
            , orderTotal = $("#orderTotal").val();
        ;
        $(this).addClass("disabled").attr("data-text", $self.html()).html("正在支付");
        if ($("#cbPreDeposit").is(':checked') == "checked" || $("#cbPreDeposit").is(':checked') == true) {
            $.ajax({
                type: "POST",
                url: "/ashx/cn/order_pay.ashx",
                data: "orderNo=" + $("#hdfOrderNo").val() + "&pd=1",
                success: function (obj) {
                    var orderTotal = parseFloat(obj.data);
                    if (orderTotal >= 0) {
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/order_pay.ashx",
                            data: "orderNo=" + $("#hdfOrderNo").val() + "&payment=" + encodeURI("预存款"),
                            success: function (data) {
                                if (data.status == 1001) {
                                    art.dialog.tips("您的余额不足以支付订单金额！<a href=\"/mobi/cn/member/pay/online.html\" style=\"color:Red; font-weight:bold;\">立即充值</a>", delay);
                                }
                                if (data.status == 1002) {
                                    art.dialog.tips("你的订单已经支付，请不要重复支付！", delay);
                                } else if (data.status == 1000) {
                                    art.dialog.tips("恭喜您，支付成功！", delay);
                                    $.ajax({
                                        type: "POST",
                                        url: "/ashx/cn/async_send_message.ashx",
                                        data: "orderNo=" + $("#hdfOrderNo").val() + "&sendType=2",
                                        global: false,
                                        async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                                        success: function () {
                                            location.href = "/mobi/cn/member/buy_gang/order/" + $("#hdfOrderNo").val() + "_info.html";
                                        }
                                    });
                                    setTimeout(function () {
                                        location.href = "/mobi/cn/member/buy_gang/order/" + $("#hdfOrderNo").val() + "_info.html";
                                    }, delay * 1000);
                                }
                            }
                        });
                    } else {
                        orderTotal = orderTotal * (-1);
                        payment($("input[name=pay_bank]:checked").val(), $("#hdfOrderNo").val(), {
                            hdfMicroMessenger: $("#hdfMicroMessenger").val(),
                            hdfWxpay: $("#hdfWxpay").val()
                        }, null, orderTotal, function (error) {
                            $self.removeClass("disabled").html($self.attr("data-text"));
                        });
                    }
                }
            });
            return false;
        } else {
            payment($("input[name=pay_bank]:checked").val(), $("#hdfOrderNo").val(), {
                hdfMicroMessenger: $("#hdfMicroMessenger").val(),
                hdfWxpay: $("#hdfWxpay").val()
            }, null, orderTotal, function (error) {
                $self.removeClass("disabled").html($self.attr("data-text"));
            });
        }
        if ($("input[name=pay_bank]:checked").attr("rel") == "货到付款") {
            location.href = "/mobi/cn/member/buy_gang/order/" + $("#hdfOrderNo").val() + ".html";
        }
        if ($("input[name=pay_bank]:checked").attr("rel") == "预存款") {
            art.dialog.tips("正在提交数据，请稍候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
            $.ajax({
                type: "POST",
                url: "/ashx/cn/order_pay.ashx",
                data: "orderNo=" + $("#hdfOrderNo").val() + "&payment=" + encodeURI("预存款"),
                success: function (msg) {
                    if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                    if (msg.status == 1001) {
                        art.dialog.tips("您的余额不足以支付订单金额！<a href=\"/mobi/cn/member/pay/online.html\" style=\"color:Red; font-weight:bold;\">立即充值</a>", delay);
                    }
                    if (msg.status == 1002) {
                        art.dialog.tips("您的订单已经支付，请不要重复支付！", delay);
                    } else if (msg.status == 1000) {
                        art.dialog.tips("恭喜您，支付成功！", delay);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/async_send_message.ashx",
                            data: "orderNo=" + $("#hdfOrderNo").val() + "&sendType=2",
                            global: false,
                            async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                            success: function () {
                                location.href = "/mobi/cn/member/buy_gang/order/" + $("#hdfOrderNo").val() + "_info.html";
                            }
                        });
                        setTimeout(function () {
                            location.href = "/mobi/cn/member/buy_gang/order/" + $("#hdfOrderNo").val() + "_info.html";
                        }, delay * 1000);
                    }
                }
            });
        }
        return false;
    });
});


//==========商品伙拼=========
$(function () {
    $("span[name='spSv']").click(function () {
        var animateTime = 500;
        $(this).addClass('cur').siblings().removeClass('cur');
        var thisid = $(this).data('attrid');
        $('.PsHpAttrList li').each(function (index, element) {
            var otherid = $(this).data('attrid');
            if (otherid != thisid) {
                $(this).hide(animateTime);
            } else {
                $(this).show(animateTime);
            }
        });

    });
    $("input[name='buyNumber']").change(function () {
        var sum = 0;
        var total = 0;
        $("input[name='buyNumber']").each(function () {
            sum += parseInt($(this).val());
        });
        var least = $("#hdLeast").val().split(",");
        var price = $("#hdPrice").val().split(",");
        for (var i = 0; i < least.length; i++) {
            if (sum >= parseInt(least[i])) {
                total = (parseFloat(price[i]) * sum);
            }
        }
        //if (total == 0) total = "低于最小起批量！";
        if (sum < parseInt(least[0])) total = "低于最小起批量！";
        $(".hpAttrChoice").addClass('hpAttrChoiceH').html("<span>" + sum + "件</span>|<span>" + total + "</span>");
    });
    $(".buyGangBuyNow").click(function () {
        var t = $(this)
            , delay = 5
        ;
        if (!t.hasClass("disabled")) {
            var gId = "",
                gC = "",
                bN = 0;
            $("input[name='buyNumber']").each(function () {
                if (parseInt($(this).val()) != 0) {
                    bN += parseInt($(this).val());
                    if (gId == "")
                        gId += $(this).attr("id");
                    else
                        gId += ("," + $(this).attr("id"));
                    if (gC == "")
                        gC += $(this).val();
                    else
                        gC += ("," + $(this).val());
                }
            });
            if (bN > 0) {
                t.addClass("disabled");
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/add_buy_gang_shopping_cart.ashx",
                    data: "gId=" + gId + "&gC=" + gC,
                    success: function (msg) {
                        if (msg == "1002") art.dialog.tips("添加购物车失败，商品库存不足！", delay);
                        else {
                            if (t.hasClass("ChBoxAddCartBt")) {
                                $(".ChoiceBoxClose").trigger("click");
                                art.dialog.tips("添加成功", delay);
                            } else location.href = "/mobi/cn/buy_gang/shopping/cart.html";

                        }
                    }
                });
                setTimeout(function () {
                    $(".buyGangBuyNow").removeClass("disabled");
                }, delay * 1000);
            } else {
                art.dialog.tips("购买数量必须为大于0的整数！", delay);
                return false;
            }
        }
    });
});

/***************** 设置帐号 ********************/
$(function () {
    var delay = 1.5;
    ///用户
    $("#setaccountForm").submit(function () {
        ///用户名
        if ($("#txtUserName").val() == "") {
            $("#txtUserName").focus();
            art.dialog.tips("请输入用户名！", delay);
            return false;
        } else {
            if ($("#txtUserName").val().length < 1 || $("#txtUserName").val().length > 16) {
                $("#txtUserName").select();
                art.dialog.tips("用户名长度不符！", delay);
                return false;
            }
        }
        ///用户密码
        if ($("#txtUserPwd").val() == "") {
            $("#txtUserPwd").focus();
            art.dialog.tips("请输入密码！", delay);
            return false;
        } else {
            if ($("#txtUserPwd").val().length < 6 || $("#txtUserPwd").val().length > 16) {
                $("#txtUserPwd").select();
                art.dialog.tips("密码长度不符，必须在6-16个字符之间！", delay);
                return false;
            }
        }
        ///确认密码
        if ($("#txtAgainPwd").val() == "") {
            $("#txtAgainPwd").focus();
            art.dialog.tips("请输入确认密码！", delay);
            return false;
        } else {
            if ($("#txtAgainPwd").val() != $("#txtUserPwd").val()) {
                $("#txtAgainPwd").select();
                art.dialog.tips("确认密码不一致！", delay);
                return false;
            }
        }
        $.ajax({
            type: "POST",
            url: "/ashx/mobi/setAccount.ashx",
            data: "userName=" + encodeURI($("#txtUserName").val()) + "&userPwd=" + encodeURI($("#txtUserPwd").val()),
            success: function (msg) {
                //                if (msg == "1002") {
                //                    $("#txtCode").select();
                //                    art.dialog.tips("验证码错误！", delay);
                //                    $("#imgCheckCode").attr("src", "/code/" + parseInt(1000 * Math.random()) + "_register.html");

                //                }
                //                else
                if (msg == "1003") {
                    $("#txtUserName").select();
                    art.dialog.tips("用户名已存在！", delay);
                } else if (msg == "1000") {
                    art.dialog.tips("恭喜您，设置成功！", delay);
                    //                        $.ajax({
                    //                            type: "POST",
                    //                            url: "/ashx/mobi/cn/simplicity/async_register_message.ashx",
                    //                            data: "userName=" + escape($("#txtUserName").val()),
                    //                            global: false,
                    //                            async: true, //jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                    //                            success: function () {
                    //                            }
                    //                        });
                    //setTimeout("location.href='/mobi/cn/member/'", delay * 1000);
                    setTimeout("window.location='/user_exit.html';", delay * 1000);
                }
            }
        });
        return false;
    });
});

/***************** 切换帐号 ********************/
$(function () {
    var delay = 1.5;
    ///切换帐号
    $("#relatedAccountForm").submit(function () {
        ///用户名
        if ($("#txtRelatedUserName").val() == "") {
            $("#txtRelatedUserPwd").focus();
            art.dialog.tips("请输入用户名！", delay);
            return false;
        }
        ///用户密码
        if ($("#txtUserPwd").val() == "") {
            $("#txtUserPwd").focus();
            art.dialog.tips("请输入密码！", delay);
            return false;
        }

        $.ajax({
            type: "POST",
            url: "/ashx/mobi/setAccount.ashx",
            data: "RelatedUserName=" + encodeURI($("#txtRelatedUserName").val()) + "&RelatedUserPwd=" + encodeURI($("#txtRelatedUserPwd").val()),
            success: function (msg) {
                if (msg == "1001") {
                    $("#txtRelatedUserName").select();
                    art.dialog.tips("用户名或密码错误！", delay);
                } else if (msg == "1003") {
                    $("#txtRelatedUserName").select();
                    art.dialog.tips("同一用户！", delay);
                } else if (msg == "1000") {
                    //art.dialog.tips("恭喜您，设置成功！", delay);
                    //setTimeout("location.href='/mobi/cn/member/'", 3000);
                    setTimeout("window.location='/mobi/';", delay * 1000);
                }
            }
        });
        return false;
    });

    $("#switchAccount").click(function () {
        this.disabled = "disabled";
        $.ajax({
            type: "POST",
            url: "/ashx/mobi/setAccount.ashx",
            data: "act=switch",
            success: function (msg) {
                if (msg == "1001") {
                    //$("#txtRelatedUserName").select();
                    art.dialog.tips("当前无法切换！", delay);
                } else if (msg == "1000") {
                    //art.dialog.tips("恭喜您，设置成功！", delay);
                    //setTimeout("location.href='/mobi/cn/member/'", 3000);
                    setTimeout("window.location='/mobi/';", delay * 1000);
                }
            }
        });
        return false;
    });
});


//***************订单列表删除按钮******************
function orderDelete(orderNo) {
    art.dialog({
        id: 'orderDelete',
        content: '您确定要删除该订单吗？',
        lock: true,
        fixed: true,
        opacity: 0.1,
        button: [
            {
                name: '确定',
                callback: function () {
                    art.dialog.tips("正在处理，请稍候  <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 60);
                    $.ajax({
                        type: "POST",
                        url: "/api/home/deleteOrder",
                        data: "orderNo=" + orderNo,
                        success: function (msg) {
                            if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                            if (msg == "1") {
                                window.location.reload();
                            } else {
                                art.dialog.tips('删除失败');
                            }
                        }
                    });
                    return false;
                },
                focus: true
            },
            {
                name: '取消',
                callback: function () {
                    art.dialog.tips('你取消了操作');
                }
            }
        ]
    });
}

//***************订单列表删除按钮******************


//***************秒杀订单列表删除按钮******************
function orderDelete_seckill(orderNo) {
    art.dialog({
        id: 'orderDelete',
        content: '您确定要删除该订单吗？',
        lock: true,
        fixed: true,
        opacity: 0.1,
        button: [
            {
                name: '确定',
                callback: function () {
                    art.dialog.tips("正在处理，请稍候  <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 60);
                    $.ajax({
                        type: "POST",
                        url: "/api/home/seckillOrderDelete",
                        data: "orderNo=" + orderNo,
                        success: function (msg) {
                            if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();   //  关闭弹窗
                            if (msg == "1") {
                                window.location.reload();
                            } else {
                                art.dialog.tips('删除失败');
                            }
                        }
                    });
                    return false;
                },
                focus: true
            },
            {
                name: '取消',
                callback: function () {
                    art.dialog.tips('你取消了操作');
                }
            }
        ]
    });
}

//***************秒杀订单列表删除按钮******************

////============================手机验证码===========================
//=====================发送手机验证码================
$(function () {
    var delay = 1.5;
    $("#txtMobileRegister").blur(function () {
        if ($(this).val() == "") {
            art.dialog.tips("请输入手机号码！", delay);
            return false;
        } else if (!formatCheck($(this).val(), "mobile")) {
            art.dialog.tips("手机号码填写不正确！", delay);
            return false;
        }
        $("#txtMobileRegister").trigger("keyup");
    });
    $("#txtMobileRegister").keyup(function () {
        if ($(this).val() == "" || !formatCheck($(this).val(), "mobile")) {
            $("#btnMobileCode").addClass('disabled');
            return false;
        } else {
            //判断手机是否已经注册账号
            $.ajax({
                type: "POST",
                url: "/ashx/cn/mobile.ashx",
                data: "type=checkmobile&mobile=" + $("#txtMobileRegister").val(),
                success: function (data) {
                    if (data == "1000") {
                        // $("#spMobile").html("<img src=\"/images/public/images/right.gif\"/>");
                        //$("#spMobile").html("");
                        $("#btnMobileCode").removeClass('disabled');
                    } else if (data == "1002") {
                        // $("#spMobile").html("已经注册！<a href=\"/mobi/cn/login.html\">登录</a> 忘记密码？点击 <a href=\"javascript:parent.location.href='/mobi/cn/password/find.html';\">这里</a>").show();
                        art.dialog.tips("手机号码已被注册过", delay);
                        $("#btnMobileCode").addClass('disabled');
                    }
                }
            });
        }
        return false;
    });

    $("#txtMobileCode").blur(function () {
        if ($(this).val() == "") {
            art.dialog.tips(" 请输入手机验证码！");
        } else {
            //手机号验证码验证
            $.ajax({
                type: "POST",
                url: "/ashx/cn/mobile.ashx",
                data: "type=validate&mobile=" + $("#txtMobileRegister").val() + "&code=" + $(this).val(),
                success: function (data) {
                    if (data == "1000") {
                        // $("#spMobileCode").html("<img src=\"/images/public/images/right.gif\"/>");
                    } else if (data = "1002") {
                        art.dialog.tips("验证失败！");
                    } else if (data == "1003") {
                        art.dialog.tips("手机验证码已过期,请重新发送！");
                    }
                }
            });
        }
        return false;
    });
    $("#btnMobileCode").click(function () {
        if ($(this).hasClass("disabled") || $(this).hasClass("finished")) return false;
        var phone_num = $("#txtMobileRegister").val();
        if (phone_num == "") {
            art.dialog.tips("填写有效的手机号");
            return false;
        }
        //手机号验证码验证
        $.ajax({
            type: "POST",
            url: "/ashx/cn/mobile.ashx",
            data: "type=send&mobile=" + phone_num,
            success: function (data) {
                $("#txtMobileCode").removeClass("disabled");
                if (data == "1004") {
                    art.dialog.tips("验证码已发送，请稍候再试");
                } else {
                    art.dialog.tips("验证码发送成功");
                    $("#btnMobileCode").addClass("finished");
                    updateTimeLabel(60);
                }
            }
        });
    });
});

//=============================商品点赞==========================
$(function () {
    var aLike = "a-like";
    $("#" + aLike).click(function () {
        var gid = $(this).data("gid");
        $.ajax({
            type: "POST",
            url: "/goods/praise",
            data: "gid=" + gid,
            success: function (obj) {
                if (obj.status == 1002) {
                    art.dialog.tips("点赞失败！");
                } else if (obj.status == 1001) {
                    window.location.href = "/mobi/cn/login.html";
                } else if (obj.status == 1003) {
                    art.dialog.tips("您已经点赞过该商品！");
                    $("#" + aLike + " .PsFun4").addClass("PsFun4H");
                } else {
                    //$("#" + aLike).text("喜欢宝贝（"+obj.total+"）");
                    $("#" + aLike).find("span").text("点赞（" + obj.total + "）");
                    art.dialog.tips("点赞成功！");
                    $("#" + aLike + " .PsFun4").addClass("PsFun4H");
                }
            }
        });
    });
});

//=============================文章点赞==========================
$(function () {
    $("#a-article-like").click(function () {
        var aid = $(this).data("aid");

        $.ajax({
            type: "POST",
            url: "/article/praise",
            data: "aid=" + aid,
            success: function (obj) {
                if (obj.status == 1002) {
                    art.dialog.tips("点赞失败！");
                } else if (obj.status == 1001) {
                    window.location.href = "/mobi/cn/login.html";
                } else if (obj.status == 1003) {
                    art.dialog.tips("您已经点赞过该文章！");
                } else {
                    $("#sPraise").text(obj.total);
                    art.dialog.tips("点赞成功！");
                }
            }
        });
    });
});

//////////////////////////////用户信息 - 验证手机号//////////////////////////////////////
$(function () {
    var delay = 1.5;
    $("#formMemberValidMobile").submit(function () {

        //--------------手机验证码验证-----------------
        if ($("#txtMobileCode1Yuan").val() != undefined) {
            if ($("#txtMobileCode1Yuan").val() == "" && $("#txtMobile1Yuan").val() != $("#txtMobile1Yuan").attr("data-val")) {
                art.dialog.tips("请输入手机验证码！");
                return false;
            }
            if ($("#txtMobileCode1Yuan").val() != "") {
                var isValidate = false;//需要给她赋值才行
                $.ajax({
                    async: false,//同步请求,防止没通过手机验证注册
                    type: "POST",
                    url: "/ashx/cn/mobile.ashx",
                    data: "type=validate&send1yuan=1&mobile=" + $("#txtMobile1Yuan").val() + "&code=" + $("#txtMobileCode1Yuan").val(),
                    success: function (data) {
                        if (data == "1000") {
                            //$("#spMobileCode").html("<img src=\"/images/public/images/right.gif\"/>");
                            //$("#txtMobile").focus();
                            isValidate = true;
                            art.dialog.tips("验证成功！", delay);
                            setTimeout(function () {
                                location.href = "/mobi/";
                            }, 3000);
                            //reutrun true;//这里返回没用，只是当前这个success方法返回一个值，下面的代码照样会执行
                        } else if (data == "100001") {
                            isValidate = true;
                            art.dialog.tips("恭喜您验证成功，一元已经充值到您的预存款账户中！", delay);
                            setTimeout(function () {
                                location.href = document.referrer;
                            }, 6000);
                        } else if (data == "1002") {
                            art.dialog.tips("验证失败！", 3);
                            //setTimeout(function(){location.href=document.referrer;},6000);
                            isValidate = false;
                        } else if (data == "1003") {
                            art.dialog.tips("手机验证码已过期,请重新发送！", delay);
                            isValidate = false;
                        } else if (data == "1005") {
                            art.dialog.tips("无开启该手机验证功能！", delay);
                            isValidate = false;
                        }
                    }
                });
                if (!isValidate) {
                    return false;
                } else { //art.dialog.tips("验证成功，一元已经充值到预存款！");
                    //return false;
                }
            }
        }

        return false;
    });
});

//////////////////////////////一元购晒单//////////////////////////////////////
$(function () {
    var delay = 1.5;
    $("#CrowdfundOrderCommentForm").submit(function () {
        if ($("#txtTitle").val() == "") {
            $("#txtTitle").focus();
            art.dialog.tips("请输入晒单标题！", delay);
            return false;
        } else {
            if ($("#txtTitle").val().length > 50) {
                $("#txtTitle").select();
                art.dialog.tips("晒单标题不能超过50个字符！", delay);
                return false;
            }
        }
        if ($("#txtaContent").val() == "") {
            $("#txtaContent").focus();
            art.dialog.tips("请输入内容！", delay);
            return false;
        } else {
            if ($("#txtaContent").val().length > 500) {
                $("#txtaContent").select();
                art.dialog.tips("内容不能超过500个字符！", delay);
                return false;
            }
        }
        if ($("#txtCommentCode").val() == "") {
            $("#txtCommentCode").focus();
            art.dialog.tips("请输入验证码！", delay);
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/Crowdfund_comment.ashx",
            data: "id=" + $("#hdGoods").val() + "&contact=" + encodeURI($("#txtContact").val()) + "&content=" + encodeURI($("#txtaContent").val()) + "&code=" + $("#txtCommentCode").val() + "&orderNo=" + $("#hdOrderNo").val() + "&imgSrc=" + encodeURI($("#imgSrc").val()) + "&title=" + encodeURI($("#txtTitle").val()),
            success: function (msg) {
                if (msg == "1003") {
                    window.location.href = "/mobi/cn/login.html";
                } else if (msg == "1002") {
                    $("#txtCommentCode").select();
                    art.dialog.tips("验证码错误，请重新输入！", delay);
                    $("#imgCheckCommentCode").attr("src", "/code/" + parseInt(10000 * Math.random()) + "_CFcomment.html");
                } else if (msg == "1001") {
                    art.dialog.tips("很抱歉，晒单失败！", delay);
                } else if (msg == "1004") {
                    art.dialog.tips("很抱歉，该订单已晒单！", delay);
                } else {
                    art.dialog.tips("恭喜您，晒单提交成功，我们会尽快核对后安排显示！", delay);
                    $('#CrowdfundOrderCommentForm')[0].reset(); //表单重置
                    setTimeout(function () {
                        window.location.href = "/mobi/cn/member/Crowdfund_winOrder/_0.html";
                    }, delay * 1000);
                    // $("#imgCheckCommentCode").attr("src", "/code/" + parseInt(10000 * Math.random()) + "_CFcomment.html");
                }
            }
        });
        return false;
    });
});

//------------------手机验证------------
$(function () {
    $("#txtMobile1Yuan").keyup(function () {
        if ($(this).val() != "" && $(this).val() == $(this).attr("data-val")) return;
        if ($(this).val() == "") {
            art.dialog.tips("请输入手机号码！");
            $("#btnMobileCode1Yuan").attr('disabled', true);
            $("#p-mobile-code").hide();
        } else {
            if (!formatCheck($(this).val(), "mobile")) {
                //art.dialog.tips("手机号码填写不正确！");
                $("#btnMobileCode1Yuan").attr('disabled', true);
                $("#p-mobile-code").hide();
            } else {
                //判断手机是否已经注册账号
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/mobile.ashx",
                    data: "type=checkmobile&mobile=" + $("#txtMobile1Yuan").val(),
                    success: function (data) {
                        if (data == "1000") {
                            //$("#spMobile").html("<img src=\"/images/public/images/right.gif\"/>");
                            $("#btnMobileCode1Yuan").attr('disabled', false);
                            $("#p-mobile-code").show();
                        } else if (data == "1002") {
                            //art.dialog.tips("已经注册,验证通过该手机号将绑定到此帐号！", delay);
                            art.dialog.tips("该手机号码已经注册了！");
                            $("#btnMobileCode1Yuan").attr('disabled', false);
                            $("#p-mobile-code").show();
                        }
                    }
                });
            }
        }
        return false;
    });

    $("#txtMobile1Yuan").blur(function () {
        if ($(this).val() != "" && $(this).val() == $(this).attr("data-val")) return;
        if ($(this).val() == "") {
            art.dialog.tips("请输入手机号码！");
        } else {
            if (!formatCheck($(this).val(), "mobile"))
                art.dialog.tips("手机号码填写不正确！");
            else {
                //判断手机是否已经注册账号
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/mobile.ashx",
                    data: "type=checkmobile&mobile=" + $("#txtMobile1Yuan").val(),
                    success: function (data) {
                        if (data == "1000") {
                            $("#spMobile").html("<img src=\"/images/public/images/right.gif\"/>");
                            $("#btnMobileCode1Yuan").attr('disabled', false);
                        } else if (data == "1002") {
                            art.dialog.tips("该手机号码已经注册了");
                            $("#btnMobileCode1Yuan").attr("disabled", "disabled");
                        }
                    }
                });
            }
        }
        return false;
    });
    //=====================发送手机验证码================
    $("#btnMobileCode1Yuan").click(function () {
        if ($(this).hasClass("disabled") || $(this).hasClass("finished")) return false;
        //手机号验证码验证
        $.ajax({
            type: "POST",
            url: "/ashx/cn/mobile.ashx",
            data: "type=send&mobile=" + $("#txtMobile1Yuan").val(),
            success: function (data) {
                $("#txtMobileCode1Yuan").attr("disabled", false);
                if (data == "1004") {
                    art.dialog.tips("验证码已发送，请稍候再试！");
                } else {
                    //调用方法
                    $("#btnMobileCode1Yuan").addClass("finished");
                    updateTimeLabel(60);
                }
            }
        });
    });
});

//==============删除收货地址=================
$(function () {
    var delay = 1.5;
    $(".AddressDeleteBt").click(function () {
        var id = $(this).data('id');
        var src = $(this).data('src');

        if (id != '') {
            $.ajax({
                type: "POST",
                url: "/ashx/cn/shopping_address.ashx",
                data: "type=del&did=" + id,
                success: function (data) {
                    if (data.status == 1001) art.dialog.tips("删除收货地址失败！", delay);
                    else if (data.status == 1000) {
                        art.dialog.tips("删除收货地址成功！", delay)
                        setTimeout(function () {
                            window.location.href = src;
                        }, 2000);
                    } else {
                        art.dialog.tips(data.msg, delay);
                    }
                }
            });
            return true;
        }
    });
});
//2016.3.26
$(function () {
    //  获取参数
    function getQueryString(name) {
        var r = window.location.search.substr(1).match("(^|&)" + name + "=([^&]*)(&|$)", "i");
        if (r != null) return decodeURIComponent(r[2]);
        return null;
    }

    // 安全问题
    $("#btnQuestionSubmit").click(function () {
        if ($("#selectQuestion").val() != "0") {
            /*if ($("#oldAnswer").val() == "") {
             $("#oldAnswer").select();
             $("#spAnswer").html("<img src=\"/images/public/images/wrong.gif\" />请输入答案！");
             return false;
             }*/
            if ($("#txtAnswer").val() == "") {
                $("#txtAnswer").select();
                $("#spAnswer").html("<img src=\"/images/public/images/wrong.gif\" />请输入答案！");
                return false;
            } else $("#spAnswer").html("<img src=\"/images/public/images/right.gif\"/>");
            $.ajax({
                type: "POST",
                url: "/ashx/cn/user_password.ashx",
                //data: "oldAnswer=" + $("#oldAnswer").val() + "&question=" + encodeURI($("#selectQuestion").val()) + "&answer=" + encodeURI($("#txtAnswer").val()),
                data: ($("#oldAnswer").length == 0 ? "" : "oldAnswer=" + $("#oldAnswer").val() + "&") + "question=" + encodeURI($("#selectQuestion").val()) + "&answer=" + encodeURI($("#txtAnswer").val()),
                success: function (obj) {
                    switch (obj.status) {
                        case 1000:
                            $("#spAnswer").html("恭喜您，更新成功！");
                            //setTimeout("window.location='/mobi/cn/center/member/user/safety.html'", 2000);
                            setTimeout("window.location='/mobi/cn/member/security/issue.html'", 2000);
                            break;
                        case 1003:
                            $("#spAnswer").html("原问题回答错误");
                            break;
                        case 1004:
                        case 1005:
                            $("#spAnswer").html("很抱歉，更新失败！");
                            break;
                    }
                }
            });
        } else $("#spAnswer").html("选择1个新的安全问题设置答案");
        return false;
    });
    // 发送消息
    $("#btnSendMessageSubmit").click(function () {
        if ($("input[name='radioType']:checked").val() == "0") {
            if ($("#txtAddressee").val() == "") {
                $("#txtAddressee").select();
                $("#spAddressee").html("<img src=\"/images/public/images/wrong.gif\" />请输入用户名！");
                return false;
            } else {
                $("#spAddressee").html("<img src=\"/images/public/images/right.gif\"/>");
            }
        } else $("#spAddressee").html("");
        if ($("#txtTitle").val() == "") {
            $("#txtTitle").select();
            $("#spTitle").html("<img src=\"/images/public/images/wrong.gif\" />请输入标题！");
            return false;
        } else $("#spTitle").html("");
        if ($("#txtaContent").val() == "") {
            $("#txtaContent").select();
            $("#spContent").html("<img src=\"/images/public/images/wrong.gif\" />请输入内容！");
            return false;
        } else $("#spContent").html("");
        var addressee = "";
        if ($("input[name='radioType']:checked").val() == "0") addressee = $("#txtAddressee").val();
        else addressee = "管理员";

        $.ajax({
            type: "POST",
            url: "/ashx/cn/send_message.ashx",
            data: "addressee=" + addressee + "&title=" + encodeURI($("#txtTitle").val()) + "&content=" + encodeURI($("#txtaContent").val()) + "&option=" + $("input[name='radioOption']:checked").val(),
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        $("#spContent").text("发送成功");
                        setTimeout("window.location.href='/mobi/cn/member/box/out/1.html'", 1000);
                        break;
                    case 1001:
                        $("#txtAddressee").select();
                        $("#spAddressee").text("不允许发送给自己！");
                        return false;
                        break;
                    case 1002:
                        $("#txtAddressee").select();
                        $("#spAddressee").text("用户名不存在！");
                        return false;
                        break;
                    case 1003:
                        $("#spContent").text("发送失败");
                        break;
                    case 1004:
                        $("#spContent").text("保存成功");
                        setTimeout("window.location.href='/mobi/cn/member/box/draft/1.html'", 1000);
                        break;
                    case 1005:
                        $("#spContent").text("保存失败");
                        break;
                    default:
                        break;
                }
            }
        });
        return false;
    });
    $("input[name='radioType']").change(function () {
        if ($("input[name='radioType']:checked").val() == "0") {
            $("#pAddressee").show();
        } else $("#pAddressee").hide();
    });
    //  发件箱/收件箱
    $(".user-inbox, .out-box").on("click", ".btn-email-detail", function () {
        //展开详情
        if (!$(".user_inbox").hasClass("hide")) {
            $(".user_inbox").addClass("hide");
            $(".user_msg_detail").removeClass("hide");
            $("#d-title").text($(this).find(".title").text());
            $("#d-from-name").text($(this).find(".author").text());
            $("#d-to-name").text($(this).find(".d-to-name").text());
            $("#d-time").text($(this).find(".time").text());
            $("#d-content").text($(this).find(".detail").text());
            $(".HeadM").attr("data-text", $(".HeadM").text());
            $(".HeadM").text("邮件详情");
            $("#btn-reply").attr("href", $(this).attr("href"));
            $(".btn-confirm").attr("data-id", $(this).attr("data-id"));
            //标为已读
            if ($(this).find(".point").length > 0) {
                var _this = $(this);
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/read_in_box.ashx",
                    data: "id=" + $(this).attr("data-id"),
                    success: function (data) {
                        switch (data.status) {
                            case 1000:
                                _this.find(".point").remove();
                                break;
                        }
                    }
                });
            }
            $(".HeadL.floatl a").click(function (event) {
                if ($(".user_inbox").hasClass("hide")) {
                    $(".user_inbox").removeClass("hide");
                    $(".user_msg_detail").addClass("hide");
                    event.preventDefault();
                    $(".HeadL.floatl a").unbind("click");
                    $(".HeadM").text($(".HeadM").attr("data-text"));
                }
            });
            event.preventDefault();
        }
    });
    //  删除邮件
    //删除全部
    $(".btn-delete-all").click(function () {
        var url = $("body").hasClass("out-box") ? "/outbox/delete" : "/inbox/delete";
        $('.user_inbox .list li').each(function (index) {
            var id = $(this).find(".btn-email-detail").attr("data-id");
            $.ajax({
                type: "POST",
                url: url,
                data: "id=" + id,
                success: function (data) {
                    switch (data.status) {
                        case 1000:
                            if (index == $('.user_inbox .list li').length - 1) {
                                $(".user_inbox .list").html('<div class="MenberNewRecListNo">全部邮件都已删除！</div>');
                            }
                            break;
                        case 1001:
                            //$(".tip-msg").text("删除失败！");
                            break;
                        default:
                            //$(".tip-msg").text("data.msg");
                            break;
                    }
                }
            });
        });
    });
    //单邮件删除
    $(".btn-delete").click(function () {
        $(".btn-delete").addClass("hide");
        $(".btn-cancel").removeClass("hide");
        $(".btn-confirm").removeClass("hide");
        $(".tip-msg").text("确定要删除？");
    });
    //取消删除
    $(".btn-cancel").click(function () {
        $(".btn-delete").removeClass("hide");
        $(".btn-cancel").addClass("hide");
        $(".btn-confirm").addClass("hide");
        $(".tip-msg").text("");
    });
    //确认删除
    $(".btn-confirm").click(function () {
        var id = $(this).attr("data-id");
        var url = $("body").hasClass("out-box") ? "/outbox/delete" : "/inbox/delete";
        if (id != '') {
            $(".tip-msg").text("正在删除");
            $.ajax({
                type: "POST",
                url: url,
                data: "id=" + id,
                success: function (data) {
                    switch (data.status) {
                        case 1000:
                            $(".tip-msg").text("删除成功！");
                            setTimeout(function () {
                                window.location.reload();
                            }, 1000);
                            break;
                        case 1001:
                            $(".tip-msg").text("删除失败！");
                            break;
                        default:
                            $(".tip-msg").text("data.msg");
                            break;
                    }
                }
            });
            return true;
        }
    });
    // 已读未读分类
    $("#NewStatus").change(function () {
        var li = $('.list li');
        switch (this.value) {
            case "0":
                li.removeClass("hide");
                break;
            case "1":
                li.each(function () {
                    if ($(this).hasClass("read")) this.addClass("hide");
                    else $(this).removeClass("hide");
                });
                break;
            case "2":
                li.each(function () {
                    if ($(this).hasClass("read")) $(this).removeClass("hide");
                    else $(this).addClass("hide");
                });
                break;
            default:
                break;
        }
    });
    //   找回密码  图片验证码:搬到user.js
    $("#imgCheckCode").click(function () {
        $(this).attr("src", '/code/' + parseInt(10000 * Math.random()) + '_findPwd.html');
    });
    //  找回密码 步骤1：确认用户名和验证码:搬到user.js
    $("#btnFindPwd").click(function () {
        var delay = 1.5;
        ///用户名
        if ($("#txtUserName").val() == "") {
            $("#txtUserName").select();
            $(".tip-msg").text("请输入用户名！");
            return false;
        }
        ///验证码
        if ($("#txtCode").val() == "") {
            $("#txtCode").select();
            $(".tip-msg").text("请输入验证码！");
            return false;
        } else $(".tip-msg").text("正在确认用户名");
        $.ajax({
            type: "POST",
            url: "/ashx/cn/find_password.ashx",
            data: "userName=" + encodeURI($("#txtUserName").val()) + "&code=" + encodeURI($("#txtCode").val()),
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        $(".passwordForm").addClass("hide");
                        if (obj.data == "email") {
                            //    验证邮箱
                            $(".passwordForm2").removeClass("hide");
                            $("#emailAddress").text(obj.email.replace(/[\w\W]{2}@/, "**@"));
                        } else {
                            //    验证安全问题
                            $(".passwordForm3").removeClass("hide");
                            $("#span-question").html(obj.data);
                            $("#btnAnswerQuestion").click(function () {
                                if ($("#answer").val() == "") {
                                    art.dialog.tips("请回答安全问题！", delay);
                                    $("#answer").focus();
                                    return false;
                                } else {
                                    $.ajax({
                                        type: "POST",
                                        url: "/ashx/cn/find_password.ashx",
                                        data: "userName=" + escape($("#txtUserName").val()) + "&answer=" + escape($("#answer").val()),
                                        success: function (data) {
                                            if (data.status == 1001) {
                                                art.dialog.tips("很抱歉，回答错误！", delay);
                                                $("#answer").select();
                                                return false;
                                            } else {
                                                location.href = "/mobi/cn/password/update.html?data=" + data.data;
                                                return false;
                                            }
                                        }
                                    });
                                }
                                return false;
                            });
                        }
                        break;
                    case 1001:
                        //验证码错误
                        break;
                    case 1002:
                        //用户名不存在
                        break;
                }
            }
        });
    });
    //    找回密码 步骤2：发送邮件
    $("#btn-send-email").click(function () {
        var delay = 1.5;
        //正在发送邮件，请稍候
        $.ajax({
            type: "POST",
            url: "/ashx/cn/find_password.ashx",
            global: false,
            async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
            data: "userName=" + encodeURI($("#txtUserName").val()),
            success: function (obj) {
                if (obj.status == 1000) {
                    //发送邮件成功
                    $(".jumper").removeClass("hide");
                    setTimeout("window.location.href='/mobi/cn/login.html'", delay * 1000);
                } else {
                    //发送邮件失败，请刷新后重试
                    art.dialog.tips("发送邮件失败，请刷新后重试！", delay);
                }
            }
        });
        return false;
    });
    //    提交评论
    $("#formOrderScore").submit(function () {
        var i
            , delay = 5
        ;
        if ($("#txtaContent").val() == "") {
            $("#txtaContent").select();
            $("#spContent").attr("placeholder", "在此输入您的评论");
            return false;
        }
        if ($("#txtaContent").val().length > 500) {
            $("#txtaContent").select();
            $(".comment-msg .tip").text('评论内容不能超过500个字符！');
            return false;
        }
        if ($("#txtCommentCode").val() == "") {
            $("#txtCommentCode").select();
            $(".code-msg .tip").text('请输入验证码！');
            return false;
        }
        var hasnopf = false;
        var hiddens = $(".shop-rating input:hidden");
        for (i = 0; i < hiddens.length; i++) {
            var val = hiddens.eq(i).val();
            if (val == "undefined" || val == '') {
                hasnopf = true;
                break;
            }
        }
        if (hasnopf) {
            art.dialog.tips("您还有没打分的选项！", delay);
            return false;
        }
        if ($("#txtCommentCode").val() == "") {
            $("#txtCommentCode").focus();
            art.dialog.tips("请输入验证码！", delay);
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/cn/goods_comment.ashx",
            data: "id=" + $("#hdGoods").val() + "&orderNo=" + $("#hdOrderNo").val() + "&score=" + $("#stars1-input").val() + "&content=" + $("#txtaContent").val() + "&imgSrc=" + "&code=" + $("#txtCommentCode").val(),
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        $(".update-tip").text("恭喜您，评论成功！");
                        if (window.location.href.indexOf("PM_") >= 0) {
                            window.setTimeout(function () {
                                window.location.href = "/mobi/cn/center/member/order/_1.html";
                            }, 2000);
                        } else if (window.location.href.indexOf("BG_") >= 0) {
                            window.setTimeout(function () {
                                window.location.href = "/mobi/cn/center/member/buy_gang/order/_1.html";
                            }, 2000);
                        }
                        break;
                    case 1001:
                        window.setTimeout(function () {
                            window.location.href = "/mobi/cn/login.html";
                        }, 2000);
                        break;
                    case 1002:
                        $("#imgCheckCommentCode").attr("src", "/code/" + parseInt(10000 * Math.random()) + "_conmment.html");
                        $("#txtCommentCode").attr("placeholder", "验证码错误");
                        $("#txtCommentCode").val("");
                        $("#txtCommentCode").select();
                        break;
                    case 1003:
                        $(".update-tip").text("评论失败，请稍后再试");
                        break;
                    default:
                        break;
                }
            }
        });
        return false;
    });
    //  更换验证码
    $("#imgCheckCommentCode,#btn-change-comment-code").click(function () {
        $("#imgCheckCommentCode").attr("src", '/code/' + parseInt(10000 * Math.random()) + '_conmment.html');
    });
    //    退款退货
    $(".upload .select-upload .add").click(function () {
        $("#filePictures").trigger("click");
    });
    //申请退款 填写银行账号
    $("input[name='refundType']").change(function () {
        if ($("input[name='refundType']:checked").val() == "1") {
            $("#divReturnBankInfo").show();
            $("#divReturnAccount").hide();
        } else if ($("input[name='refundType']:checked").val() == "2" || $("input[name='refundType']:checked").val() == "3") {
            $("#divReturnBankInfo").hide();
            $("#divReturnAccount").show();
        } else {
            $("#divReturnBankInfo").hide();
            $("#divReturnAccount").hide();
        }
        $(".tip").text("");
    });


    function checkFilePictures() {
        if (formatCheck($("#filePictures").val(), "image")) {
            $(".tip").text("");
            $(this).parent().find(".img-title").text("已选择");
            return true;
        } else {
            $(".tip").text("凭证只能上传gif, png, jpg, bmp格式的图片！");
            $("#filePictures").focus();
            return false;
        }
    }

    $("#filePictures").change(checkFilePictures);

    //申请退款 表格提交
    $("#formRefundSubmit").click(function () {
        if ($("#RefundReason").val() == "") {
            $("#RefundReason").focus();
            $("#RefundReason").attr("placeholder", "请输入退款原因！");
            return false;
        }
        if ($("#filePictures").val() != "" && $("#filePictures").val() != undefined) {
            if (!checkFilePictures()) {
                return false;
            }
        }
        if ($("input[name='refundType']:checked").val() == "1") {
            if ($("#txtRefundBank").val() == "") {
                $("#txtRefundBank").focus();
                $(".tip").text("请输入退款银行！");
                return false;
            }
            if ($("#txtBankCard").val() == "") {
                $("#txtBankCard").focus();
                $(".tip").text("请输入银行卡号！");
                return false;
            }
            if ($("#txtAccountName").val() == "") {
                $("#txtAccountName").focus();
                $(".tip").text("请输入开户姓名！");
                return false;
            }
        } else if ($("input[name='refundType']:checked").val() == "2" || $("input[name='refundType']:checked").val() == "2") {
            if ($("#txtAccount").val() == "") {
                $("#txtAccount").focus();
                $(".tip").text("请输入退款账户！");
                return false;
            }
        }
        $('#formRefund').ajaxSubmit({
            success: function (data) {
                var delay = 1.5 //  延迟时间
                    , orderNo = document.getElementById("hdOrderNo").value  //  订单号
                    , orderInfoHref //  订单详情链接
                ;
                switch (data.status) {
                    case 1000:
                        switch (true) {
                            case orderNo.indexOf("MS_") > -1:
                                orderInfoHref = "/mobi/cn/member/seckill/order/" + orderNo + ".html";
                                break;
                            case orderNo.indexOf("PM_") > -1:
                            default:
                                orderInfoHref = "/mobi/cn/member/order/" + orderNo + "_info.html";
                                break;
                        }
                        art.dialog.tips(data.msg, delay);
                        setTimeout("location.href = '" + orderInfoHref + "'", delay * 1000);
                        break;
                    case 1002:
                        art.dialog.tips(data.msg, delay);
                        break;
                    case 1003:
                        art.dialog.tips(data.msg, delay);
                        break;
                }
            }
        });
    });
    //    消费记录 筛选跳转
    $(".user_expense_records .filter-select").change(function () {
        var link = $(this).val();
        if (link) {
            window.location.href = link;
        }
    });
    //判断当前页面是哪个分类，激活样式
    $(".filter-category a").each(function () {
        if (window.location.href.indexOf($(this).attr("href")) > 0) {
            $(this).addClass("main-color");
            return false;
        }
    });
    $(".filter-progress .progress").each(function () {
        if (window.location.href.indexOf($(this).val()) > 0) {
            $(this).attr("selected", true);
        }
    });
    //  删除草稿箱邮件
    $(".draft .icon-delete-2").click(function () {
        var id = $(this).attr("data-id");
        if (id != '') {
            $.ajax({
                type: "POST",
                url: "/draftbox/delete",
                data: "id=" + id,
                success: function (data) {
                    switch (data.status) {
                        case 1000:
                            $(".draft .icon-delete-2").each(function () {
                                if (id == $(this).attr("data-id")) {
                                    var target = $(this).parent().parent();
                                    target.animate({"opacity": "0"}, 1000);
                                    setTimeout(function () {
                                        target.remove();
                                    }, 1000);
                                    return false;
                                }
                            });
                            break;
                        case 1001:
                            alert("删除失败");
                            break;
                        default:
                            alert(data.msg);
                            break;
                    }
                }
            });
            return false;
        }
    });
    //   会员信息 获取微信头像
    $("#btn-reflash-avatar").click(function () {
        $("#frmImg").attr("src", "/ashx/mobi/wxGZHAuth.ashx?act=refreshHeadImg");
        return false;
    });
    //  分佣中心 推广赚佣金 下拉
    $(".commission_index .btn-drop-down").click(function () {
        if ($(this).next(".content-drop-down").hasClass("hide")) {
            $(".commission_index .btn-drop-down").each(function () {
                $(this).next(".content-drop-down").addClass("hide");
            })
            $(this).next(".content-drop-down").removeClass("hide");
        } else {
            $(this).next(".content-drop-down").addClass("hide");
        }
        return false;
    })
    //  打开图片地址
    $('.click-to-open').click(function () {
        if ($(this).attr("src")) location.href = $(this).attr("src");
    });
    //  获取推广二维码和海报
    $(".btn_Weixin_first").click(function () {  //  首次生成需要
        if ($("#Weixin_QR_LIMIT_SCENE_con").attr("hidden") || $("#Big_Weixin_QR_LIMIT_SCENE_con").attr("hidden")) {
            $("#Weixin_QR_LIMIT_SCENE_con,#Big_Weixin_QR_LIMIT_SCENE_con").removeAttr("hidden");
            $(".btn_Weixin_QR_LIMIT_SCENE").trigger("click");
        }
    });
    $(".btn_Weixin_QR_LIMIT_SCENE").click(function () {     //  生成
        $(this).attr("disabled", true);
        $(".content-drop-down .tip").text("正在生成");
        $.ajax({
            url: "/ashx/get_Weixin_QR_LIMIT_SCENE.ashx?type=commission",
            dataType: "text",
            async: true,
            success: function (msg) {
                switch (msg) {
                    case "nomembercard":
                        location.href = "/ashx/mobi/wxGZHAuth.ashx?act=picSend";
                        break;
                    case "-1":
                        $(".content-drop-down .tip").text("创建失败，请稍后再试");
                        break;
                    case "0":
                        $(".content-drop-down .tip").text("暂时无法创建，请稍后重试");
                        break;
                    default:
                        if (msg.indexOf(".jpg") > 0) {
                            $(".content-drop-down .tip").text("");
                            $("#Weixin_QR_LIMIT_SCENE_con").attr("src", msg + '?' + (new Date()));
                            $("#Big_Weixin_QR_LIMIT_SCENE_con").attr("src", msg.replace('.jpg', '_Big.jpg?') + (new Date()));
                        }
                        break;
                }
                var Weixin_XCX_LIMIT_SCENE_con = $("#Weixin_XCX_LIMIT_SCENE_con").attr("src");
                var Big_Weixin_XCX_LIMIT_SCENE_con = $("#Big_Weixin_XCX_LIMIT_SCENE_con").attr("src");
                $("#Weixin_QR_LIMIT_SCENE_con").attr("src", Weixin_XCX_LIMIT_SCENE_con + "?" + (new Date()));
                $("#Big_Weixin_XCX_LIMIT_SCENE_con").attr("src", Big_Weixin_XCX_LIMIT_SCENE_con + "?" + (new Date()));

                $('#Weixin_QR_LIMIT_SCENE_con').removeAttr("disabled");
            }
        });
        return true;
    });

    $(".btn_XCX_QR_LIMIT_SCENE").click(function () {     //  生成
        $(this).attr("disabled", true);
        $(".content-drop-down .tip").text("正在生成");
        $.ajax({
            url: "/ashx/get_XCX_QR_LIMIT_SCENE.ashx",
            dataType: "text",
            async: true,
            success: function (msg) {
                switch (msg) {
                    case "0":
                        $(".content-drop-down .tip").text("暂时无法创建，请稍后重试");
                        break;
                    default:
                        $(".content-drop-down .tip").text("生成成功！");
                        break;
                }
                var Weixin_XCX_LIMIT_SCENE_con = $("#Weixin_XCX_LIMIT_SCENE_con").attr("src");
                var Big_Weixin_XCX_LIMIT_SCENE_con = $("#Big_Weixin_XCX_LIMIT_SCENE_con").attr("src");
                $("#Weixin_XCX_LIMIT_SCENE_con").attr("src", Weixin_XCX_LIMIT_SCENE_con + "?" + (new Date()));
                $("#Big_Weixin_XCX_LIMIT_SCENE_con").attr("src", Big_Weixin_XCX_LIMIT_SCENE_con + "?" + (new Date()));

                $('#Weixin_XCX_LIMIT_SCENE_con').removeAttr("disabled");
            }
        });
        return true;
    });

    //  申请成为微推客
    $("#popedomApplyBtn").click(function () {
        if ($("#txtapplyWord").val() != undefined) {
            if ($("#txtapplyWord").val() == "") {
                art.dialog.tips("请输入" + commission_ApplyWordTitle + "！");
                return false;
            }
        }

        $(this).attr("disabled", true);
        $.ajax({
            url: "/ashx/user_commissionPopedom.ashx?type=apply" + ($("#txtapplyWord").val() != undefined ? "&applyWord=" + escape($("#txtapplyWord").val()) : ""),
            async: true,
            dataType: "json",
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        alert('已成功提交申请。');
                        $('#popedomStateCon').text('当前状态:申请中');
                        break;
                    case 1002:
                        alert('无法申请，请与管理员联系。');
                        break;
                    default:
                        alert('暂时无法创建，请稍后重试。');
                        $("#popedomApplyBtn").removeAttr("disabled");
                        break;
                }
            }
        });
    });
    //分佣申请提现
    $("#formCommissionApply").submit(function () {
        var delay = 1.5;
        var errstr = "";
        if ($("#commissionpwd").val() == "") {
            errstr = "请填写提现密码";
        } else if ($('input[name="withdrawalType"]:checked').val() == "1") {
            $(".MenberRFytxXj").find("input").each(function () {
                if ($(this).val() == "") {
                    errstr = "请先填写银行账户信息";
                }
            })
        }
        if (errstr != "") {
            art.dialog.tips(errstr, delay);
            return false;
        }
        $.ajax({
            type: "post",
            url: "/commission/Withdrawal",
            dataType: "JSON",
            data: "commissionpwd=" + $("#commissionpwd").val() + "&amount=" + $("#amount").val() + "&bankinfo=" + $("#bankinfo").val() + "&PayeeBank=" + $("#PayeeBank").val() + "&PayeeAccount=" + $("#PayeeAccount").val() + "&PayeeFullName=" + $("#PayeeFullName").val() + "&withdrawalType=" + $('input[name="withdrawalType"]:checked').val() + "&id=" + $("#payRecord").val(),
            success: function (data) {
                switch (data.status) {
                    case 1000:
                        art.dialog.tips("提交成功", delay);
                        setTimeout(function () {
                            location.href = "/mobi/cn/member/commission/WithdrawalHistory/1.html";
                        }, 1500);
                        break;
                    case 1001:
                        art.dialog.tips("当前不允许修改！", delay);
                        break;
                    case 1002:
                        art.dialog.tips("提现密码不正确！", delay);
                        break;
                    case 1004:
                        art.dialog.tips(data.msg, delay);//"必须使用微信才可以提现到微信零钱！"有几个返回这一错误号
                        break;
                    case 1005:
                        //art.dialog.tips("您的会员帐号未绑定微信号！", delay);
                        art.dialog.confirm("您的会员帐号未绑定微信号！是否将您当前的微信号与当前的会员帐号绑定", function () {
                            $.ajax({
                                type: "post",
                                url: "/commission/Withdrawal",
                                dataType: "JSON",
                                data: "commissionpwd=" + $("#commissionpwd").val() + "&amount=" + $("#amount").val() + "&bankinfo=" + $("#bankinfo").val() + "&PayeeBank=" + $("#PayeeBank").val() + "&PayeeAccount=" + $("#PayeeAccount").val() + "&PayeeFullName=" + $("#PayeeFullName").val() + "&withdrawalType=" + $('input[name="withdrawalType"]:checked').val() + "&id=" + $("#payRecord").val() + "&bindWX=1",
                                success: function (data) {
                                    switch (data.status) {
                                        case 1000:
                                            art.dialog.tips("您的当前会员帐号成功绑定当前微信号，提现申请成功提交！", delay);
                                            setTimeout(function () {
                                                location.href = "/mobi/cn/member/commission/WithdrawalHistory/1.html";
                                            }, delay * 1000);
                                            break;
                                        case 1001:
                                            art.dialog.tips("当前不允许修改！", delay);
                                            break;
                                        case 1002:
                                            art.dialog.tips("提现密码不正确！", delay);
                                            break;
                                        case 1008:
                                            art.dialog.tips("会员未绑定微信，且当前微信号已绑定另一会员帐号，无法绑定当前会员帐号！", delay);
                                            break;
                                        default:
                                            art.dialog.tips("提交失败,请稍后重试！", delay);
                                            break;
                                    }
                                },
                                error: function () {
                                    art.dialog.tips("提交超时", delay);
                                }
                            });
                            return true;
                        }, function () {
                            art.dialog.tips('你取消了操作');
                        });
                        break;
                    case 1006:
                        art.dialog.tips("必须使用会员原始绑定的微信号才可以提现到微信零钱！", delay);
                        break;
                    case 1007:
                        art.dialog.tips("会员微信绑定不属于当前微信公众号！", delay);
                        break;
                    default:
                        art.dialog.tips("提交失败,请稍后重试！", delay);
                        break;
                }
            },
            error: function () {
                art.dialog.tips("提交超时", delay);
            }
        });
        return false;
    });
    //佣金返现方式：银行转账、预存款
    $(".withdrawalType").click(function () {
        var val = $(this).val();
        if (val == "1") $('.MenberRFytxXj').show();
        if (val == "2") $('.MenberRFytxXj').hide();
    });
    //    找回分佣提现密码
    $("#formAnswerQuestionCommission").submit(function () {
        if ($("#answer").val() == "") {
            art.dialog.tips("请回答安全问题！", delay);
            $("#answer").focus();
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/cn/find_password.ashx",
            data: "userName=" + encodeURI($("#txtUserName").val()) + "&answer=" + encodeURI($("#answer").val()),
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        location.href = "/mobi/cn/commission/password/update.html?t=2&data=" + obj.data;
                        break;
                    case 1001:
                        art.dialog.tips("很抱歉，回答错误！", delay);
                        $("#answer").select();
                        break;
                }
            }
        });
        return false;
    });
    //    重置佣金提现密码
    $("#formUpdateCommissionPwd").submit(function () {
        var delay = 1.5;
        //  新密码
        if ($("#txtUserPwd").val() == "") {
            $("#txtUserPwd").select();
            $("#spPwd").html("请输入密码！");
            return false;
        } else if ($("#txtUserPwd").val().length < 6 || $("#txtUserPwd").val().length > 16) {
            $("#txtUserPwd").select();
            $("#spPwd").html("密码必须为6-16个字符");
            return false;
        } else $("#spPwd").html("<img src=\"/images/public/images/right.gif\"/>");
        ///确认密码
        if ($("#txtAgainPwd").val() == "") {
            $("#txtAgainPwd").select();
            $("#spAgPwd").html("请再输入一遍新密码");
            return false;
        } else if ($("#txtAgainPwd").val() != $("#txtUserPwd").val()) {
            $("#txtAgainPwd").select();
            $("#spAgPwd").html("两次输入的密码不一致！");
            return false;
        } else $("#spAgPwd").html("<img src=\"/images/public/images/right.gif\"/>");
        $.ajax({
            type: "POST",
            url: "/ashx/user_CommissionPassword.ashx",
            dataType: 'json',
            data: "data=" + getQueryString("data") + "&t=2&newPwd=" + encodeURI($("#txtUserPwd").val()),
            success: function (data) {
                switch (data.status) {
                    case 1000:
                        art.dialog.tips("密码更新成功！", delay);
                        setTimeout("window.location.href='/mobi/cn/member/commission/index.html'", delay * 1000);
                        break;
                    default:
                        art.dialog.tips("密码更新失败，请刷新重试！", delay);
                        break;
                }
            }
        });
        return false;
    });
    //帮助中心搜索
    $('#helpSearch').click(function () {
        location.href = "/mobi/cn/help/list.html?kw=" + encodeURIComponent($("#helpSearchTxt").val()) + "";
        return false;
    });
    //    商品搜索
    $("#goods-search").click(function () {
        location.href = "/mobi/cn/goods_list.html?kw=" + $("#goods-search-kw").val().trim();
        return false;
    });
    //    设置佣金提现密码
    $("#btnCommissionPwdSubmit").click(function () {
        var delay = 1.5;
        if ($("#txtOldCommissionPwd").val() == "") {
            $("#txtOldCommissionPwd").select();
            return false;
        }
        if ($("#txtNewCommissionPwd").val().length < 6 || $("#txtNewCommissionPwd").val().length > 16) {
            $("#txtNewCommissionPwd").select();
            return false;
        }
        if ($("#txtAgainCommissionPwd").val() == "") {
            $("#txtAgainCommissionPwd").select();
            return false;
        } else if ($("#txtAgainCommissionPwd").val() != $("#txtNewCommissionPwd").val()) {
            $("#txtAgainCommissionPwd").select();
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/user_Commissionpassword.ashx",
            data: "oldPwd=" + $("#txtOldCommissionPwd").val() + "&newPwd=" + $("#txtNewCommissionPwd").val(),
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        art.dialog.tips("恭喜您，佣金提现密码设置成功！", delay);
                        if (document.referrer.indexOf("/mobi/cn/member/commission/index.html") > 0 || document.referrer.indexOf("/mobi/cn/member/commission/Withdrawal.html") > 0) setTimeout(function () {
                            top.location.href = document.referrer;
                        }, 2000);
                        else
                            setTimeout("window.location='/mobi/cn/member/'", delay * 1000);
                        break;
                    case 1001:
                    case 1005:
                        art.dialog.tips("很抱歉，佣金提现密码设置失败！", delay);
                        break;
                    case 1002:
                        $("#txtOldCommissionPwd").select();
                        $("#spOldCommissionPwd").html("<img src=\"/images/public/images/wrong.gif\" />原密码错误！");
                        break;

                }
            }
        });
        return false;
    });
    //    修改用户密码
    $("#btnPwdSubmit").click(function () {
        if ($("#txtOldPwd").val() == "") {
            $("#txtOldPwd").select();
            return false;
        }
        if ($("#txtNewPwd").val().length < 6 || $("#txtNewPwd").val().length > 16) {
            $("#txtNewPwd").select();
            return false;
        }
        if ($("#txtAgainPwd").val() == "") {
            $("#txtAgainPwd").select();
            return false;
        } else if ($("#txtAgainPwd").val() != $("#txtNewPwd").val()) {
            $("#txtAgainPwd").select();
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/cn/user_password.ashx",
            data: "oldPwd=" + $("#txtOldPwd").val() + "&newPwd=" + $("#txtNewPwd").val(),
            success: function (obj) {
                var delay = 1.5;
                switch (obj.status) {
                    case 1000:
                        art.dialog.tips("恭喜您，更新成功！", delay);
                        setTimeout("window.location='/mobi/cn/member/'", delay * 1000);
                        break;
                    case 1001:
                        $("#txtOldPwd").select();
                        art.dialog.tips("原密码错误", delay);
                    case 1005:
                    case 1002:
                        art.dialog.tips("很抱歉，更新失败！", delay);
                        break;

                }
            }
        });
        return false;
    });
    //    积分转预存款
    $("#btnPoint_Turn").click(function () {
        var errstr = "", url;
        if (errstr != "") {
            art.dialog.tips(errstr, delay);
            return false;
        }
        url = location.href;
        $.ajax({
            type: "post",
            url: url,
            data: "yuanAmount=" + $('select[name="yuanAmount"] option:selected').val(),
            dataType: "json",
            success: function (data) {
                switch (data.status) {
                    case 1000:
                        art.dialog.tips("积分转换成功！", delay);
                        setTimeout(function () {
                            location.reload();
                        }, 1500);
                        break;
                    case 1001:
                        art.dialog.tips("当前不允许修改！", delay);
                        break;
                    case 1002:
                        art.dialog.tips("提现密码不正确！", delay);
                        break;
                    default:
                        art.dialog.tips("提交失败,请稍后重试！", delay);
                        break;
                }
            },
            error: function () {
                art.dialog.tips("提交超时", delay);
            }
        });
        return false;
    });
    //    预存款充值
    $("#btnPay").click(function () {

        var oId = "0",
            payType = $("input[name='pay_bank']:checked").attr("data-name"),
            delay = 5,
            $self = $(this);

        if ($("#txtAlimoney").val() == "") {
            $("#txtAlimoney").focus();
            art.dialog.tips("请输入充值金额！", delay);
            return false;
        } else {
            if ($("#txtAlimoney").val().match(/^\d*\.?\d{0,2}$/) == null) {
                $("#txtAlimoney").select();
                art.dialog.tips("充值金额不符，最多保留两位小数！", delay);
                return false;
            }
        }
        if ($("input[name='pay_bank']:checked").val() == null) {
            $("input[name='pay_bank']")[0].focus();
            art.dialog.tips("请选择支付方式！", delay);
            return false;
        }
        $(this).addClass("disabled").attr("data-text", $self.val()).val("正在支付");
        $.ajax({
            type: "POST",
            url: "/ashx/cn/online_pay.ashx",
            data: "expenseMoney=" + $("#txtAlimoney").val() + "&payType=" + payType,
            global: true,
            async: false,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
            success: function (msg) {
                if (msg.status == 1000) {
                    oId = msg.data.orderNo;
                } else return false;
            },
            error: function () {
                alert("获取订单号错误");
            }
        });
        payment($("input[name='pay_bank']:checked").val(), oId, {
            hdfWxpay: $("#hdfWxpay").val(),
            hdfMicroMessenger: $("#hdfMicroMessenger").val()
        }, $("#txtAlimoney").val(), null, function (error) {
            $self.removeClass("disabled").html($self.attr("data-text"));
        });

        return false;
    });
    //    方便点击充值方式
    $(".RechargePayTypeList img").click(function () {
        $(this).parent().find(".radio").trigger("click");
    });
    //开启抵用券转赠
    $("#btnCouponPass").click(function () {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/coupon/online/open",
            data: "coupon=" + $("#hdCouponCode").val(),
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        art.dialog.tips("开启转赠成功！", delay);
                        window.setTimeout(function () {
                            art.dialog.tips("开启转赠成功！", delay);
                            window.history.go(0);
                        }, 1500);
                        break;
                    case 1001:
                        art.dialog.tips(obj.msg, delay);
                        break;
                    case 1002:
                        art.dialog.tips("该抵用券已经使用过了!", delay);
                        break;
                    case 1003:
                        art.dialog.tips("该抵用券不在有效期范围内!", delay);
                        break;
                    case 1004:
                        art.dialog.tips("没有满足该抵用券使用金额!", delay);
                        break;
                    case 1004:
                        art.dialog.tips("开启转赠失败请稍后再试!", delay);
                        break;
                }
            }
        });
    });
    //取消抵用券转赠
    $("#btnCouponCancelPass").click(function () {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/coupon/online/cancel",
            data: "coupon=" + $("#hdCouponCode").val(),
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        art.dialog.tips("取消转赠成功！", delay);
                        window.setTimeout(function () {
                            art.dialog.tips("取消转赠成功！", delay);
                            window.history.go(0);
                        }, 1500);
                        break;
                    case 1001:
                        art.dialog.tips(obj.msg, delay);
                        break;
                    case 1002:
                        art.dialog.tips("该抵用券已经使用过了!", delay);
                        break;
                    case 1003:
                        art.dialog.tips("该抵用券不在有效期范围内!", delay);
                        break;
                    case 1004:
                        art.dialog.tips("没有满足该抵用券使用金额!", delay);
                        break;
                    case 1004:
                        art.dialog.tips("取消转赠失败请稍后再试!", delay);
                        break;
                }
            }
        });
    });

    //=====================使用抵用券================
    //    抵用券下拉框发生变化
    $("#selectCoupon").change(function () {
        var deliveryMoney = 0.0
            , delay = 5
        ;
        if ($("#hdFromFreight").val() == "false")
            if ($("input[name='deliveryWay']").prop("checked")) {
                deliveryMoney = parseFloat($("input[name='deliveryWay']").val().split(',')[1]);
            }
        var value = $(this).val();
        if (value == "0") return;
        //抵用券后的订单信息,这里是正常显示的
        $.ajax({
            type: "POST",
            url: "/coupon/online/use",
            data: "coupon=" + value + "&deliveryMoney=" + deliveryMoney + "&geid=" + $("#hdGoodsGeid").val() + "&amount=" + $("#hdBuyNowAmount").val(),
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        $(".p-coupon").show();
                        $("#hdCoupon").val(value);
                        $("#hdCouponParValue").val(obj.data.parValue);
                        $(".p-coupon-class").html(obj.data.name);
                        $("#p-card-info").hide();
                        $("#sSumTotal").text(changePrice2money(parseFloat(obj.data.needFee).toFixed(2)));
                        //$("#hdGoodsSum").val(obj.data.needFee);
                        break;
                    case 1001:
                        art.dialog.tips(obj.msg, delay);
                        break;
                    case 1002:
                        art.dialog.tips("该抵用券已经使用过了!", delay);
                        break;
                    case 1003:
                        art.dialog.tips("该抵用券不在有效期范围内!", delay);
                        break;
                    case 1004:
                        art.dialog.tips("没有满足该抵用券使用金额!", delay);
                        break;
                }
            }
        });
    });
    //取消使用抵用券
    $("#btnCancelCoupon").click(function () {
        var deliveryMoney = 0.0;
        if ($("#hdFromFreight").val() == "false")
            if ($("input[name='deliveryWay']").prop("checked")) {
                deliveryMoney = parseFloat($("input[name='deliveryWay']").val().split(',')[1]);
            }
        var value = $("#hdCoupon").val();
        var parValue = parseFloat($("#hdCouponParValue").val());
        var goodsSum = parseFloat($("#hdGoodsSum").val());
        if (!value) return;
        $(".p-coupon").show();
        $("#hdCoupon").val('');
        $("#hdCouponParValue").val('0');
        $("#p-card-info").hide();
        //$("#a-input-coupon").show();
        //$("#hdGoodsSum").val(goodsSum + parValue);
        $("#sSumTotal").text(changePrice2money(parseFloat(goodsSum + deliveryMoney).toFixed(2)));
        $("#hdCoupon").val('');
        $(".p-coupon").hide();
        $(".p-coupon-class").html('');
        $("#p-card-info").show();
        $("#selectCoupon").val('0');
        $("#span-input").show();
        $("#txtCoupon").val('');
    });
    //点击直接使用抵用券
    /*$("#a-input-coupon").click(function () {
     $(this).hide();
     $("#span-input").show();
     });*/
    //点击确认使用抵用券按钮
    $("#btnCoupon").click(function () {
        $(".a-coupon").text("使用");
        var deliveryMoney = 0.0;
        if ($("#hdFromFreight").val() == "false")
            if ($("input[name='deliveryWay']").prop("checked")) {
                deliveryMoney = parseFloat($("input[name='deliveryWay']").val().split(',')[1]);
            }
        var coupon = $("#txtCoupon").val().trim();
        if (coupon == '') return;
        //使用抵用券后的订单信息
        $.ajax({
            type: "POST",
            url: "/coupon/online/use",
            data: "&coupon=" + coupon + "&deliveryMoney=" + deliveryMoney + "&geid=" + $("#hdGoodsGeid").val() + "&amount=" + $("#hdBuyNowAmount").val(),
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        $(".p-coupon").show();
                        $("#sSumTotal").text(changePrice2money(parseFloat(obj.data.needFee).toFixed(2)));
                        $("#hdCoupon").val(coupon);
                        $(".p-coupon-class").html(obj.data.name);
                        $("#p-card-info").hide();
                        //$("#a-input-coupon").show();
                        $("#span-input").hide();
                        $("#txtCoupon").val('');
                        //$("#hdGoodsSum").val(obj.data.needFee);
                        $("#hdCouponParValue").val(obj.data.parValue);
                        break;
                    case 1001:
                        art.dialog.tips(obj.msg, delay);
                        break;
                    case 1002:
                        art.dialog.tips("该抵用券已经使用过了!", delay);
                        break;
                    case 1003:
                        art.dialog.tips("该抵用券不在有效期范围内!", delay);
                        break;
                    case 1004:
                        art.dialog.tips("没有满足该抵用券使用金额!", delay);
                        break;
                }
            }
        });
    });
    //抵用券线下使用
    $("#btnCouponUse").click(function () {
        var delay = 1.5;
        ///用户名
        if ($("#couponUserName").val() == "") {
            $("#couponUserName").focus();
            art.dialog.tips("请输入管理账号！", delay);
            return false;
        }
        ///用户密码
        if ($("#couponUserPwd").val() == "") {
            $("#couponUserPwd").focus();
            art.dialog.tips("请输入密码！", delay);
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/coupon/downline/use",
            data: "coupon=" + $("#hdCouponCode").val() + "&admin=" + $("#couponUserName").val() + "&password=" + $("#couponUserPwd").val(),
            success: function (obj) {
                if (obj.status != 1000) {
                    art.dialog.tips(obj.msg, delay);
                    $("#couponUserName").focus();
                } else {
                    art.dialog.tips("使用成功！", delay);
                    window.setTimeout(function () {
                        art.dialog.tips("使用成功！", delay);
                        window.history.go(0);
                    }, delay * 1000);
                }
            }
        });
    });

    // 立即结算-普通订单
    $("#aImmediatelySettle,.aImmediatelySettle").click(function () {
        var delay = 1.5;
        if ($("#txtCount").val() == "0") {
            art.dialog.tips("请输入购买数量！", delay);
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/cn/detection_session.ashx",
            success: function (obj) {
                switch (obj.status) {
                    case 1001:
                        location.href = "/mobi/cn/login.html";
                        break;
                    case 1000:
                    case 1002:
                        //location.href = "/mobi/cn/order/submit/" + $('#hdGoodsEntitys').val() + "/" + $("#txtCount").val() + ".html";
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/settlement.ashx",
                            data: "goodsEntitys=" + $("#hdGoodsEntitys").val() + "&count=" + $("#txtCount").val(),
                            success: function (obj) {
                                switch (obj.status) {
                                    case 1001:
                                        location.href = "/mobi/cn/login.html";
                                        break;
                                    case 1002:
                                        art.dialog.tips("商品库存不足！", delay);
                                        break;
                                    case 1003:
                                        art.dialog.tips("超过限购数量，请购买其他商品", delay);
                                        break;
                                    default:
                                        location.href = "/mobi/cn/order/submit/" + $('#hdGoodsEntitys').val() + "/" + $("#txtCount").val() + ".html";
                                        break;
                                }
                            }
                        });
                        //$.ajax({
                        //    type: "POST",
                        //    url: "/ashx/cn/add_shopping_cart.ashx",
                        //    data: "goodsEntitys=" + $("#hdGoodsEntitys").val() + "&count=" + $("#txtCount").val(),
                        //    success: function (obj) {
                        //        switch (obj.status) {
                        //            case 1000:
                        //                location.href = "/mobi/cn/order/submit/" + $("#hdGoodsEntitys").val() + ".html";
                        //                break;
                        //            case 1001:
                        //                // art.dialog.tips("请登陆后再进行购买！", delay);
                        //                location.href = "/mobi/cn/login.html";
                        //                break;
                        //            case 1002:
                        //                art.dialog.tips("商品库存不足！", delay);
                        //                break;
                        //            case 1003:
                        //                art.dialog.tips("超过限购数量，请购买其他商品", delay);
                        //                break;
                        //            default:
                        //                break;
                        //        }
                        //    }
                        //});
                        break;
                    default:
                        break;
                }
            }
        });
        return false;
    });


    // 立即结算-拼团订单
    $("#aImmediatelySettle_spell,.aImmediatelySettle_spell").click(function () {
        var delay = 1.5;
        if ($("#txtCount").val() == "0") {
            art.dialog.tips("请输入购买数量！", delay);
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/cn/detection_session.ashx",//判断有没有登录
            success: function (obj) {
                switch (obj.status) {
                    case 1001:
                        location.href = "/mobi/cn/login.html";
                        break;
                    case 1000:
                    case 1002:
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/add_shopping_cart_spell.ashx",
                            data: "goodsEntitys=" + $("#hdGoodsEntitys").val() + "&count=" + $("#txtCount").val() + "&joinGroupNo=" + $("#joinGroupNo").val(),
                            success: function (obj) {
                                switch (obj.status) {
                                    case 1000:
                                        //拼团订单提交
                                        location.href = "/mobi/cn/spellOrder/submit/" + $("#hdGoodsEntitys").val() + ".html";
                                        break;
                                    case 1001:
                                        art.dialog.tips("请登陆后再进行购买！", delay);
                                        location.href = "/mobi/cn/login.html";
                                        break;
                                    case 1002:
                                        art.dialog.tips("商品库存不足！", delay);
                                        break;
                                    case 1003:
                                        art.dialog.tips("超过限购数量，请购买其他商品", delay);
                                        break;
                                    case 10031:
                                        art.dialog.tips("本商品您已经拼团过" + obj.hasbuy + "件,还可以拼团" + obj.canbuy + "件", delay);
                                        break;
                                    case 10032:
                                        art.dialog.tips("很抱歉,本商品的拼团活动已经结束,无法购买.", delay);
                                        setTimeout("parent.location.reload()", delay * 1000);
                                        break;
                                    case 10033:
                                        art.dialog.tips("该产品规格不参与拼团活动!", 3);
                                        break;
                                    case 10034:
                                        art.dialog.tips("您已经参与此商品的拼团了!", 2.5);
                                        setTimeout("location.href='/mobi/cn/group_spell/" + obj.spellNo + ".html'", 3 * 1000);
                                        break;
                                    case 10035:
                                        art.dialog.tips("抱歉,此团已满人,请开团或者选择其它的团!", 2.5);
                                        setTimeout("location.href='/mobi/cn/spell/" + obj.goodid + ".html'", 3 * 1000);
                                        break;
                                    case 10036:
                                        art.dialog.tips("抱歉,此团已结束,请开团或者选择其它的团!", 2.5);
                                        setTimeout("location.href='/mobi/cn/spell/" + obj.goodid + ".html'", 3 * 1000);
                                        break;
                                    case 10037:
                                        setTimeout("location.href='/mobi/cn/member/spell/order/" + obj.orderNo + "_info.html'", 1000);
                                        break;
                                    default:
                                        break;
                                }
                            }
                        });
                        break;
                    default:
                        break;
                }
            }
        });
        return false;
    });


    // 立即结算-手机版秒杀
    $("#aImmediatelySettle_seckill,.aImmediatelySettle_seckill").click(function () {
        var delay = 1.5;
        if ($("#txtCount").val() == "0") {
            art.dialog.tips("请输入购买数量！", delay);
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/cn/detection_seckill_session.ashx",
            success: function (data) {
                switch (data.status) {
                    case 1001:
                        //未登录,去登录
                        location.href = "/mobi/cn/login.html";
                        break;
                    case 1000:  //  正常结算,没有break,继续到1002执行
                    case 1002:  // 正常结算
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/add_shopping_cart_seckill.ashx",
                            data: "addType=buynow&goodsEntitys=" + $("#hdGoodsEntitys").val() + "&count=" + $("#txtCount").val(),
                            dataType: "json",
                            success: function (obj) {
                                switch (obj.status) {
                                    case 1000:
                                        location.href = "/mobi/cn/seckill/order/submit/" + $('#hdGoodsEntitys').val() + ".html";
                                        break;
                                    case 1001:
                                        //art.dialog.tips("请登陆后再进行购买！", delay);
                                        location.href = "/mobi/cn/login.html";
                                        break;
                                    case 1002:
                                        art.dialog.tips("商品库存不足！", delay);
                                        break;
                                    case 1003:
                                        art.dialog.tips("您已经买过" + obj.buycount + "件了,给别的亲留一点吧", delay);
                                        break;
                                    case 10031:
                                        art.dialog.tips("您买过" + obj.buycount + "件,还可以买" + obj.suggestBuy + "件", delay);
                                        break;
                                    default:
                                        break;
                                }
                            }
                        });
                        break;
                    default:
                        break;
                }
            }
        });
        return false;
    });
    ///用户登录
    $("#formLogin").submit(function () {
        var delay = 1.5;
        ///用户名
        if ($("#txtUserName").val() == "") {
            art.dialog.tips("请输入用户名！", delay);
            $("#txtUserName").focus();
            return false;
        } else {
            document.getElementById("txtUserName").value = document.getElementById("txtUserName").value.replace(/\s/g, "");
        }
        ///用户密码
        if ($("#txtUserPwd").val() == "") {
            art.dialog.tips("请输入用户密码！", delay);
            $("#txtUserPwd").focus();
            return false;
        }
        ///验证码
        if ($("#txtCode").val() == "") {
            art.dialog.tips("请输入验证码！", delay);
            $("#txtCode").focus();
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/api/user/UserLogin",
            data: "frommobile=true&userName=" + $("#txtUserName").val() + "&userPwd=" + encodeURI($("#txtUserPwd").val()) + "&code=" + encodeURI($("#txtCode").val()) + "&saveDay=" + $("#selectSave").val(),
            success: function (data) {
                switch (data.status) {
                    case 1000:
                        art.dialog.tips("登录成功,正在跳转！", delay);
                        if (getUrlParam("redirect")) {
                            setTimeout("location.href='" + getUrlParam("redirect") + "'", delay * 1000);
                        } else if (document.referrer != "" && document.referrer.indexOf("/mobi/cn/password/update.html") === -1 && document.referrer.indexOf("/mobi/cn/register.html") === -1 && document.referrer.indexOf("/mobi/cn/login.html") === -1) {
                            setTimeout("location.href=document.referrer;", delay * 1000);
                        } else {
                            setTimeout('location.href = "/mobi"', delay * 1000);
                        }
                        break;
                    case 1001:
                        $("#txtCode").select();
                        art.dialog.tips("验证码错误，请重新输入！", delay);
                        break;
                    case 1002:
                        $("#txtUserPwd").select();
                        art.dialog.tips("密码错误！", delay);
                        break;
                    case 1003:
                        $("#txtUserName").select();
                        art.dialog.tips("帐号已被停用！", delay);
                        break;
                    case 1004:
                        $("#txtUserName").select();
                        art.dialog.tips("该用户还没有通过审核，请耐心等待或联系管理员！", delay);
                        break;
                    case 1005:
                        $("#txtUserName").select();
                        art.dialog.tips("用户名不存在！", delay);
                        break;
                    default:
                        break;
                }
                $("#imgCheckCode").trigger("click");
            }
        });
        return false;
    });
    //  秒杀删除所选商品-秒杀
    $("#btn-delete-cart-selected-seckill").click(function () {
        if ($(".CartProChecked.cur").length > 0) {
            art.dialog({
                id: 'testID',
                content: "确定要删除选择的商品吗？",
                lock: true,
                fixed: true,
                opacity: 0.1,
                button: [
                    {
                        name: '确定',
                        callback: function () {
                            var gData = [],
                                sData = [],
                                g = 'gpid=',
                                s = 'sid=';
                            $(".CartProChecked.cur").each(function () {
                                if ($(this).attr("data-packid")) {
                                    gData.push(g + $(this).attr("data-packid"));
                                    if (g == 'gpid=') g = ",";
                                } else {
                                    sData.push(s + $(this).attr("data-sid"));
                                    if (s == 'sid=') s = ",";
                                }
                            });
                            var data = gData.length ? gData.join("") : "" + sData.length ? ("&" + sData.join("")) : "";
                            if (data != "") changeShoppingCart_seckill(data);
                        },
                        focus: true
                    },
                    {
                        name: '取消',
                        callback: function () {
                        }
                    }
                ]
            });
        }
    });
    //  删除所选商品
    $("#btn-delete-cart-selected").click(function () {
        if ($(".CartProChecked.cur").length > 0) {
            art.dialog({
                id: 'testID',
                content: "确定要删除选择的商品吗？",
                lock: true,
                fixed: true,
                opacity: 0.1,
                button: [
                    {
                        name: '确定',
                        callback: function () {
                            var gData = [],
                                sData = [],
                                g = 'gpid=',
                                s = 'sid=';
                            $(".CartProChecked.cur").each(function () {
                                if ($(this).attr("data-packid")) {
                                    gData.push(g + $(this).attr("data-packid"));
                                    if (g == 'gpid=') g = ",";
                                } else {
                                    sData.push(s + $(this).attr("data-sid"));
                                    if (s == 'sid=') s = ",";
                                }
                            });
                            var data = gData.length ? gData.join("") : "" + sData.length ? ("&" + sData.join("")) : "";
                            if (data != "") changeShoppingCart(data);
                        },
                        focus: true
                    },
                    {
                        name: '取消',
                        callback: function () {
                        }
                    }
                ]
            });
        }
    });
    //  购物车
    $("#edit-shoppingcart").delegate(".amountJia", "click", function () {
        //  商品数量加一
        var obj = $(this).attr("data-sid");
        var data = "cid=" + obj + "&amount=" + (parseInt($("#amount" + obj).val()) + 1);
        changeShoppingCart(data);
    }).delegate(".amountJian", "click", function () {
        //  商品数量减一
        var obj = $(this).attr("data-sid");
        if (parseInt($("#amount" + obj).val()) > 1) {
            $(this).removeClass('SCMinusH');
            var data = "cid=" + obj + "&amount=" + (parseInt($("#amount" + obj).val()) - 1);
            changeShoppingCart(data);
        }
    }).delegate(".changeAmount", "change", function () {
        //  直接修改商品数量
        var obj = $(this).attr("data-sid");
        if (isNaN(parseInt($("#amount" + obj).val())) || parseInt($("#amount" + obj).val()) <= 0) {
            art.dialog.tips("购买数量不能小于0！", 1.5);
            return false;
        } else {
            var data = "cid=" + obj + "&amount=" + $("#amount" + obj).val();
            changeShoppingCart(data);
        }
    }).delegate(".btn-delete-cart-goods", "click", function () {
        //  删除商品或套餐
        var data;
        if ($(this).attr("data-packid") && $(this).attr("data-packid") != 0) data = 'gpid=' + $(this).attr("data-packid");
        else data = 'sid=' + $(this).attr("data-sid");
        art.dialog({
            id: 'testID',
            content: '确定要删除该商品吗？',
            lock: true,
            fixed: true,
            opacity: 0.0,
            button: [
                {
                    name: '确定',
                    callback: function () {
                        changeShoppingCart(data);
                    },
                    focus: true
                },
                {
                    name: '取消',
                    callback: function () {
                    }
                }
            ]
        });
    }).delegate(".CartProChecked", "click", function () {
        //  多选勾选
        var obj = $(this).attr("data-geid")
        ;
        if (!$(this).hasClass("disabled")) {
            $('.' + obj + '').toggleClass('cur');
            cartCheckChange();
            if (!$('.' + obj + '').hasClass('cur')) {
                $('.CartProAllChecked').removeClass('cur');
            }
        }
    });
    //  清除失效商品按钮-秒杀
    $("#btn-delete-cart-goods-seckill-overtime").click(function () {
        var data = "operate=clearOvertime";
        art.dialog({
            id: 'testID',
            content: "确定要消除失效的商品吗？",
            lock: true,
            fixed: true,
            opacity: 0.1,
            button: [
                {
                    name: '确定',
                    callback: function () {
                        changeShoppingCart_seckill(data);
                    },
                    focus: true
                },
                {
                    name: '取消',
                    callback: function () {
                    }
                }
            ]
        });
    });
    //  秒杀购物车
    $("#edit-shoppingcart-seckill").delegate(".amountJia", "click", function () {
        //商品数量加一
        var obj = $(this).attr("data-sid");
        var data = "cid=" + obj + "&amount=" + (parseInt($("#amount" + obj).val()) + 1);
        changeShoppingCart_seckill(data);
    }).delegate(".amountJian", "click", function () {
        //  商品数量减一
        var obj = $(this).attr("data-sid");
        if (parseInt($("#amount" + obj).val()) > 1) {
            $(this).removeClass('SCMinusH');
            var data = "cid=" + obj + "&amount=" + (parseInt($("#amount" + obj).val()) - 1);
            changeShoppingCart_seckill(data);
        }
    }).delegate(".changeAmount", "change", function () {
        //  直接修改商品数量
        var obj = $(this).attr("data-sid");
        if (isNaN(parseInt($("#amount" + obj).val())) || parseInt($("#amount" + obj).val()) <= 0) {
            art.dialog.tips("购买数量不能小于0！", 1.5);
            return false;
        } else {
            var data = "cid=" + obj + "&amount=" + $("#amount" + obj).val();
            changeShoppingCart_seckill(data);
        }
    }).delegate(".btn-delete-cart-goods", "click", function () {
        //  删除商品或套餐
        var data;
        if ($(this).attr("data-packid") && $(this).attr("data-packid") != 0) data = 'gpid=' + $(this).attr("data-packid");
        else data = 'sid=' + $(this).attr("data-sid");
        art.dialog({
            id: 'testID',
            content: '确定要删除该商品吗？',
            lock: true,
            fixed: true,
            opacity: 0.0,
            button: [
                {
                    name: '确定',
                    callback: function () {
                        changeShoppingCart_seckill(data);
                    },
                    focus: true
                },
                {
                    name: '取消',
                    callback: function () {
                    }
                }
            ]
        });
    }).delegate(".CartProChecked", "click", function () {
        //  多选勾选
        if (!$(this).hasClass("disabled")) {
            var obj = $(this).attr("data-geid")
            ;
            $('.' + obj + '').toggleClass('cur');
            cartCheckChange();
            if (!$('.' + obj + '').hasClass('cur')) {
                $('.CartProAllChecked').removeClass('cur');
            }
        } else {
            // art.dialog.tips("不可选择！", 1.5);
        }
    });

    /**
     * 购物车控制器
     * @constructor
     */
    function CartController() {
        this.checklabels = ".CartProChecked";
    }

    /**
     * 设置值
     * @param name {string} 键名
     * @param value 键值
     */
    CartController.prototype.set = function (name, value) {
        this[name] = value;
    }
    /**
     * 获取购物车内当前的商品（只在购物车页面使用）
     * @param isSelected {Boolean|Options} 只要选中的商品/未选中的商品，不填为全部商品
     */
    CartController.prototype.getGoods = function (isSelected) {
        var resultData = {
            "goods": {}   //  商品
            , "package": {}   //  套餐
        };  //  要返回的商品信息

        $(this.checklabels).each(function () {
            var dataId = this.getAttribute("data-geid")
                , activeClass = "cur"   //  选中样式
            ;

            switch (isSelected) {
                //  只要未选中的商品
                case false:
                    if (this.hasClass(activeClass)) {
                        return true;
                    }
                //    只要选中的商品
                case true:
                    if (!this.hasClass(activeClass)) {
                        return true;
                    }
                //  其他情况
                default:
                    /*if (typeof isSelected === "function") {
                     isSelected(this);
                     return true;
                     }*/
                    break;
            }
            resultData[dataId] = this.id;
        });

        return resultData;
    };
    /**
     * 保存选择状态
     */
    CartController.prototype.saveSelectStatus = function () {
        this.selectedData = this.getGoods(true);
    };
    /**
     * 重现选择状态
     */
    CartController.prototype.showSelectStatus = function () {
        var newData = this.getGoods()
            , selectedData = this.selectedData
        ;

        for (var id in newData) {
            var selectedItem = selectedData[id] ? document.getElementById(selectedData[id]) : null;

            if (selectedItem) {
                selectedItem.click();
            }
        }
    };

    /**
     * 秒杀购物车控制器
     * @constructor
     */
    function SeckillCartController() {
    }

    SeckillCartController.prototype = new CartController();

    //  购物车，增删加减改都调这个
    function changeShoppingCart(data, bidAppend) {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/ashx/cn/shopping_cart.ashx",
            data: data,
            //  清空购物车 data='sid=all'
            dataType: "json",
            success: function (obj) {
                var cartController = new CartController();

                switch (obj.status) {
                    case 1001:
                        art.dialog.tips('您购买的商品总额最多换购' + obj.amount + '件此商品！', delay);
                        break;
                    case 1002:
                        art.dialog.tips("很抱歉，换购失败，商品库存不足！", delay);
                        break;
                    case 1003:
                        art.dialog.tips("已达到最大数量", delay);
                        break;
                    case 1004:
                        location.href = "/mobi/cn/login.html?redirect=" + location.href;
                        break;
                    case 0:  //  0是删除多个商品成功
                    case 2000:  //  2000是增减改商品数量成功
                    case 3000:  //  3000是删除商品成功
                        //  成功，用百度模板重新显示购物车内的商品信息
                        if (!bidAppend) {
                            cartController.saveSelectStatus();
                            $(".CartList").children().remove();
                            var html = baidu.template('sc-shoppingCart', obj);
                            $(".CartList").append(html);
                            // $("#CartTotalNum").html(obj.data.sumTotal);
                            $("#CartTotalNum").html(0);
                            if (typeof MainController.application.appCartChangeHandler === 'function') MainController.application.appCartChangeHandler(obj.data.sumAmount);
                            //$('.CartProAllChecked,.CartProChecked').removeClass('cur');
                            cartController.showSelectStatus();
                            cartCheckChange();
                        }
                        break;
                    case 2001:
                        art.dialog.tips("很抱歉，此操作导致商品总额不足换购商品，商品数量修改失败！", delay);
                        break;
                    case 2002:
                        art.dialog.tips("很抱歉，商品数量修改失败，商品库存不足！", delay);
                        break;
                    case 3001:
                        art.dialog.tips("很抱歉，此操作导致商品总额不足换购商品，商品删除失败！", delay);
                        break;
                    default:
                        break;
                }
            },
            error: function (obj) {
                alert("您操作得太快了，请休息一会");
            }
        });
        return false;
    }

    //  秒杀购物车，增删加减改都调这个
    function changeShoppingCart_seckill(data, bidAppend) {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/ashx/cn/seckill_shopping_cart.ashx",
            data: data,
            //  清空购物车 data='sid=all'
            dataType: "json",
            success: function (obj) {
                var seckillCartController = new SeckillCartController();

                switch (obj.status) {
                    case 1001:
                        art.dialog.tips('您购买的商品总额最多换购' + obj.amount + '件此商品！', delay);
                        break;
                    case 1002:
                        art.dialog.tips("很抱歉，换购失败，商品库存不足！", delay);
                        break;
                    case 1003:
                        art.dialog.tips("已达到最大数量", delay);
                        break;
                    case 2000:  //  2000是增减改商品数量成功
                    case 3000:  //  3000是删除商品成功
                        //  成功，用百度模板重新显示购物车内的商品信息
                        if (!bidAppend) {
                            seckillCartController.saveSelectStatus();
                            $(".CartList").children().remove();
                            var html = baidu.template('sc-shoppingCart', obj);
                            $(".CartList").append(html);
                            $("#CartTotalNum").html(obj.data.sumTotal);
                            //$('.CartProAllChecked,.CartProChecked').removeClass('cur');
                            seckillCartController.showSelectStatus();
                            cartCheckChange();
                        }
                        break;
                    case 2001:
                        art.dialog.tips("很抱歉，此操作导致商品总额不足换购商品，商品数量修改失败！", delay);
                        break;
                    case 2002:
                        art.dialog.tips("很抱歉，商品数量修改失败，商品库存不足！", delay);
                        break;
                    case 3001:
                        art.dialog.tips("很抱歉，此操作导致商品总额不足换购商品，商品删除失败！", delay);
                        break;
                    default:
                        break;
                }
            },
            error: function (obj) {
                alert("您操作得太快了，请休息一会");
            }
        });
        return false;
    }

    //  购物车 结算按钮
    $("#aSettlement").click(function () {
        var delay = 1.5;

        if (!$('.CartProChecked').hasClass('cur')) {
            art.dialog.tips("请选择商品！", delay);
            return false;
        }
        $.ajax({
            type: "POST",
            url: "/ashx/cn/detection_session.ashx",
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        var checkedGoods = [],  //  存储商品标识
                            checkName = "CartProChecked", //  选择状态标签样式名称
                            /**
                             * 添加选中商品
                             * @param target {Object} 目标节点
                             * @param attrName  {string} 属性名称
                             */
                            addGoods = function (target, attrName) {
                                checkedGoods.push(target.getAttribute(attrName));
                            }
                        ;

                        //    获取选中商品
                        $('.' + checkName + '.cur').each(function () {
                            var packIdAttrName = "data-packid"  //  套餐标识属性名称
                                , packid = $(this).attr(packIdAttrName) //  套餐标识
                                , attrName = "data-geid"    //  商品标识属性名称
                            ;

                            if (packid === "0") {
                                addGoods(this, attrName);
                            } else {
                                $('.CartLi[' + packIdAttrName + '=' + packid + ']').each(function () {
                                    addGoods(this, attrName);
                                });
                            }
                        });
                        location.href = "/mobi/cn/order/submit/" + checkedGoods.join("_") + ".html";
                        break;
                    case 1001:
                        //登录注册弹窗 显示
                        window.location = "/mobi/cn/login.html";
                        break;
                    case 1002:
                        art.dialog.tips("购物车目前没有加入任何商品！", delay);
                        break;
                    default:
                        break;
                }
            }
        });
        return false;
    });
    //  秒杀购物车 结算按钮
    $("#aSettlement_seckill").click(function () {
        var delay = 1.5;
        if (!$('.CartProChecked').hasClass('cur')) {
            art.dialog.tips("请选择商品！", delay);
            return false;
        }
        var geid = "", Pgeid = "";
        $('.CartProChecked.cur').each(function (index, element) {
            var Datageid = "", PDgeid = "";
            var packid = $(this).data('packid');
            if (packid > 0 && packid != null) {
                $('.Package' + packid + '').each(function () {
                    var PDgeid = "_" + $(this).data('geid') + "";
                    Pgeid += PDgeid;
                });
            }
            if (index == 0) {
                Datageid = $(this).attr('data-geid');
            } else {
                Datageid = "_" + $(this).attr('data-geid') + "";
            }
            geid = geid + Datageid;
        });
        $.ajax({
            type: "POST",
            url: "/ashx/cn/detection_seckill_session.ashx",
            data: "submitType=shoppingcart&geid=" + geid,
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        window.location = "/mobi/cn/seckill/order/submit/" + geid + ".html";
                        break;
                    case 1001:
                        //登录注册弹窗 显示
                        window.location = "/mobi/cn/login.html";
                        break;
                    case 1002:
                        art.dialog.tips("购物车目前没有加入任何商品！", delay);
                        break;
                    case 1003:
                        art.dialog.tips("选中的商品超过限购数,请检查！", delay);
                        //setTimeout(function () {
                        //    window.location.reload();
                        //}, 2000);
                        break;
                    default:
                        break;
                }
            }
        });
        return false;
    });
    //     伙拼购物车
    $("#abgSettlement").click(function () {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/ashx/cn/buy_gang_detection_session.ashx",
            success: function (data) {
                if (data.status == 1001)
                    window.location = "/mobi/cn/login.html";
                else if (data.status == 1002)
                    art.dialog.tips("购物车目前没有加入任何伙拼商品！", delay);
                else if (data.status == 1003)
                    art.dialog.tips("伙拼商品数量没有达到最低起批量！", delay);
                else if (data.status == 1000)
                    window.location = "/mobi/cn/buy_gang/order/submit.html";
            }
        });
        return false;
    });
    //  会员注册
    $("#btnRegisterSubmit").click(function () {
        var delay = 1.5;
        if ($("#txtUserName").val() == "") {
            art.dialog.tips('请填写用户名！', delay);
            return false;
        } else {
            document.getElementById("txtUserName").value = document.getElementById("txtUserName").value.replace(/\s/g, "");
        }
        if ($("#txtUserPwd").val() == "" || $("#txtUserPwd").val().length < 6) {
            art.dialog.tips('请填写6位密码！', delay);
            return false;
        }
        if ($("#txtAgainPwd").val() != $("#txtUserPwd").val()) {
            art.dialog.tips('前后密码不一致！', delay);
            return false;
        }
        if ($("#txtEmail").attr("rel") == "1" && $("#txtEmail").val() == "") {
            $("#txtEmail").focus();
            art.dialog.tips('请输入电子邮箱！', delay);
            return false;
        }
        if ($("#txtEmail").val().length > 0) {
            if (!/(\d|\w)+@(\d|\w)+\.(\d|\w)+/.test($("#txtEmail").val())) {
                art.dialog.tips('填写正确的邮箱', delay);
                return false;
            }
        }
        if ($("#txtMobileCode").length && $("#txtMobileCode").val() == "") {
            art.dialog.tips('请填写手机验证码！', delay);
            return false;
        }
        if (!$("#cbServer").is(':checked')) {
            art.dialog.tips('请先同意条款！', delay);
            return false;
        }
        var data = "";
        $("#formRegister input[name],#formRegister select[name]").each(function (index) {
            var and = "";
            if (index != 0) and = "&";
            data += and + $(this).attr("name") + "=" + $(this).val();
        });
        $.ajax({
            type: "POST",
            url: "/api/user/userRegister",
            data: data,
            dataType: "json",
            success: function (data) {
                switch (data.status) {
                    case 1000:
                        if (data.state == 2) {
                            art.dialog.tips('恭喜您，注册成功，请等待管理员审核！', delay);
                            if (location.href.indexOf("/mobi/cn/register.html") >= 0) {
                                setTimeout("location.href='/mobi/index.html'", delay * 1000);
                            } else {
                                art.dialog.tips('恭喜您，注册成功！', delay);
                                setTimeout("parent.location.reload()", delay * 1000);
                            }
                        } else {
                            art.dialog.tips('恭喜您，注册成功！', delay);
                            $.ajax({
                                type: "POST",
                                url: "/ashx/cn/async_register_message.ashx",        //  传用户名，发邮件
                                data: "userName=" + encodeURI($("#txtUserName").val()),
                                global: false,
                                async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                                success: function () {
                                }
                            });
                            if (location.href.indexOf("/cn/register/web.html") >= 0 || location.href.indexOf("/cn/register/web.html") >= 0) {
                                setTimeout("location.href='/mobi/index.html'", delay * 1000);
                            } else if (parent.location.href.indexOf("/cn/shopping/cart.html") >= 0) {
                                setTimeout("parent.location.reload()", delay * 1000);
                            } else {
                                setTimeout("location.href='/mobi/index.html'", delay * 1000);
                            }
                        }
                        break;
                    case 1001:
                        $("#imgCheckCode").attr("src", "/code/" + parseInt(1000 * Math.random()) + "_register.html");
                        art.dialog.tips('很抱歉，注册失败！！', delay);
                        break;
                    case 1002:
                        art.dialog.tips('验证码有误！', delay);
                        break;
                    case 1003:
                        art.dialog.tips('用户名已存在！', delay);
                        break;
                    case 1004:
                        art.dialog.tips('电子邮箱已注册！', delay);
                        break;
                    case 1009:
                        art.dialog.tips('手机验证码信息有误！', delay);
                        break;
                }
            }
        });
        return false;
    });
    //团购购物车 结算
    $("#imgBulkSettlement").click(function () {
        $.ajax({
            type: "POST",
            url: "/ashx/cn/detection_bulk_session.ashx",
            success: function (data) {
                switch (data.status) {
                    case 1000:
                        window.location = "/mobi/cn/bulk/order/submit.html";
                        break;
                    case 1001:
                        location.href = "/mobi/cn/login.html";
                        break;
                    case 1002:
                        art.dialog.tips("团购购物车目前没有加入任何商品！", 1.5);
                        break;
                }
            }
        });
        return false;
    });
    //     百度分享按钮
    if ($("#btn-bd-share-page").length) {
        MainController.weChatMiniProgram.checks()
            .done(function () {
                $("#btn-bd-share-page").parent().remove();
            })
            .fail(function () {
                window._bd_share_config = {
                    "common": {
                        "bdSnsKey": {},
                        "bdText": "",
                        "bdMini": "2",
                        "bdMiniList": false,
                        "bdPic": "",
                        "bdStyle": "0",
                        "bdSize": "16"
                    },
                    "share": {"bdSize": 16},
                    "selectShare": {
                        "bdContainerClass": null,
                        "bdSelectMiniList": ["qzone", "tsina", "tqq", "renren", "weixin"]
                    }
                };
                $("body").append('<script src="http://bdimg.share.baidu.com/static/api/js/share.js?cdnversion=' + ~(-new Date() / 36e5) + '"></script><div id="shareBox" class="bdsharebuttonbox" data-tag="share_1"><a class="bds_mshare" data-cmd="mshare"></a><a class="bds_qzone" data-cmd="qzone" href="#"></a><a class="bds_tsina" data-cmd="tsina"></a><a class="bds_baidu" data-cmd="baidu"></a><a class="bds_renren" data-cmd="renren"></a><a class="bds_tqq" data-cmd="tqq"></a><a class="bds_more" data-cmd="more">更多</a><a class="bds_count" data-cmd="count"></a></div>');
                $("#btn-bd-share-page").removeClass("disabled").click(function () {
                    /*if ($("#shareBox").hasClass("active")) {
                     $("#shareBox").removeClass("active");
                     }*/
                    // else {
                    // $("#shareBox").addClass("active");
                    // $(".bdshare_dialog_box").show();
                    document.querySelector(".bds_more").click();    //  触发百度分享的更多事件
                    // }
                    return false;
                });
                /*$("#shareBox").click(function () {
                 $(this).removeClass("active");
                 });*/
            });
    }

    /**                      一元竞购                           */
        //    云购购物车

    var $loadingCartCon;

    /**                   一元购结束             */

    //  团购详情 购买
    $("#bulkBuy").click(function () {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/ashx/cn/add_bulk_shopping_cart.ashx",
            data: "groupBuy=" + $("#hdGroupBuy").val(),
            success: function (msg) {
                switch (msg.status) {
                    case 1001:
                        window.location = "/mobi/cn/login.html";
                        break;
                    case -1002:
                        art.dialog.tips("添加购物车失败，团购已经结束！", delay);
                        break;
                    case -1003:
                        art.dialog.tips("添加购物车失败，商品库存不足！", delay);
                        break;
                    case -1004:
                        art.dialog.tips("添加购物车失败，请重试！", delay);
                        break;
                    default:
                        //  成功
                        window.location = "/mobi/cn/bulk/shopping/cart.html";
                        break;
                }
            }
        });
        return false;
    });
    //  团购购物车
    $("#bulk-cart-list").delegate(".aBulkMountJian", "click", function () {
        //减少商品数量
        var obj = $(this).attr("data-sid");
        if (parseInt($("#amount" + obj).val()) > 1) {
            var data = "cid=" + obj + "&amount=" + (parseInt($("#amount" + obj).val()) - 1);
            changeBulkShoppingCart(data);
        }
    }).delegate(".changeBulkMount", "change", function () {
        //修改商品数量
        var obj = $(this).attr("data-sid");
        $("#amount" + obj).val($("#amount" + obj).val().replace(/[^\d]+/, 0));
        if (parseInt($("#amount" + obj).val()) <= 0) {
            art.dialog.tips("购买数量不能小于0！", 1.5);
            return false;
        }
        var data = "cid=" + obj + "&amount=" + $("#amount" + obj).val();
        changeBulkShoppingCart(data);
    }).delegate(".aBulkMountJia", "click", function () {
        //增加商品数量
        var obj = $(this).attr("data-sid");
        var data = "cid=" + obj + "&amount=" + (parseInt($("#amount" + obj).val()) + 1);
        changeBulkShoppingCart(data);
    });

    //  团购购物车 增删改商品数量
    function changeBulkShoppingCart(data) {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/ashx/cn/bulk_shopping_cart.ashx",
            data: data,
            success: function (obj) {
                switch (obj.status) {
                    case 1000:
                        $(".CartList").children().remove();
                        var div = [];
                        for (var i = 0, l = obj.data.length; i < l; i++) {
                            var g = obj.data[i];
                            div[i] = '<div class="CartLi"><div class="floatl"></div><div class="floatr"><p class="CartZPrice">￥'
                                + g.presentPrice + '</p></div><div class="CartM"><div class="CartMPic"><a href="/mobi/cn/bulk/'
                                + g.gid + '.html"><span class="ImgMiddle"></span><img class="lazy" data-original="'
                                + g.pictures + '" src="' + g.pictures + '"/></a></div><div class="CartMPro"><div class="CartMName"><a href="/mobi/cn/bulk/'
                                + g.gid + '.html">' + g.groupBuyName + '</a></div><div class="NumBox"><span class="SCMinus SCMinusH aBulkMountJian" data-sid="' + g.sid + '"></span><input class="SCNumTxt changeBulkMount" type="text" value="'
                                + g.amount + '" id="amount' + g.sid + '" maxlength="5" data-sid="'
                                + g.sid + '"/><span class="SCPlus aBulkMountJia" data-sid="' + g.sid + '"></span><span class="SCMaximum" style="display:none;"></span></div></div></div></div>';
                        }
                        $(".CartList").append($(div.join("")));
                        $("#imgBulkSettlement").html("去结算(" + obj.sumAmount + ")");
                        $(".FCartMTotal").html('合计:￥' + obj.sumTotal);
                        break;
                    case 1001:
                        location.href = "/mobi/cn/login.html";
                        break;
                    case 1002:
                        art.dialog.tips("很抱歉，商品数量修改失败,商品库存不足！", delay);
                        break;
                    case 1003:
                        art.dialog.tips("很抱歉，商品数量修改失败！", delay);
                        break;
                }
            }
        })
    }


    //  秒杀商品详情 点击立即秒杀
    $("#seckillBuy").click(function () {
        var delay = 1.5;
        $.ajax({
            type: "POST",
            url: "/ashx/cn/add_seckill_shopping_cart.ashx",
            data: "seckillBuy=" + $("#hdSeckillBuy").val(),
            success: function (msg) {
                switch (msg.status) {
                    case 1001:
                        window.location = "/mobi/cn/login.html";
                        break;
                    case -1002:
                        art.dialog.tips("添加购物车失败，团购已经结束！", delay);
                        break;
                    case -1003:
                        art.dialog.tips("添加购物车失败，商品库存不足！", delay);
                        break;
                    case -1004:
                        art.dialog.tips("添加购物车失败，请重试！", delay);
                        break;
                    case -1005:
                        art.dialog.tips("您已经抢过了哟,给别的亲留一点吧。", delay);
                        break;
                    default:
                        //  成功
                        window.location = "/mobi/cn/seckill/shopping/cart.html";
                        break;
                }
            }
        });
        return false;
    });
    //秒杀购物车 结算
    $("#imgSeckillSettlement").click(function () {
        $.ajax({
            type: "POST",
            url: "/ashx/cn/detection_seckill_session.ashx",
            success: function (data) {
                switch (data.status) {
                    case 1000:
                        window.location = "/mobi/cn/seckill/order/submit.html";
                        break;
                    case 1001:
                        location.href = "/mobi/cn/login.html";
                        break;
                    case 1002:
                        art.dialog.tips("秒杀购物车目前没有加入任何商品！", 1.5);
                        break;
                }
            }
        });
        return false;
    });
    ///伙拼购物车
    $("#edit-shoppingcart").delegate(".amountbgJian", "click", function () {
        //减少商品数量
        var obj = $(this).attr("data-sid");
        if (parseInt($("#amount" + obj).val()) > 1) {
            var poststr = "cid=" + obj + "&amount=" + (parseInt($("#amount" + obj).val()) - 1);
            $.ajax({
                type: "POST",
                url: "/ashx/cn/buy_gang_shopping_cart.ashx",
                data: poststr,
                dataType: 'text',
                success: function (obj) {
                    changebgShoppingCart(obj);
                }
            });
        }
        return false;
    }).delegate(".changebgAmount", "change", function () {
        var obj = $(this).attr("data-sid");
        if (parseInt($("#amount" + obj).val()) == 0) {
            art.dialog.tips("购买数量不能为0！", 1.5);
            return false;
        }
        var poststr = "cid=" + obj + "&amount=" + $("#amount" + obj).val();

        $.ajax({
            type: "POST",
            url: "/ashx/cn/buy_gang_shopping_cart.ashx",
            data: poststr,
            success: function (obj) {
                changebgShoppingCart(obj);
            }
        });
        return false;
    }).delegate(".amountbgJia", "click", function () {
        var obj = $(this).attr("data-sid");
        var poststr = "cid=" + obj + "&amount=" + (parseInt($("#amount" + obj).val()) + 1);

        $.ajax({
            type: "POST",
            url: "/ashx/cn/buy_gang_shopping_cart.ashx",
            data: poststr,
            success: function (obj) {
                changebgShoppingCart(obj);
            }
        });
        return false;
    }).delegate(".customButtonbgShoppingCart", "click", function () {
        var sid = $(this).attr("data-sid");
        art.dialog({
            id: 'testID',
            content: '确定要删除吗？',
            lock: true,
            fixed: true,
            opacity: 0.1,
            button: [
                {
                    name: '确定',
                    callback: function () {
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/buy_gang_shopping_cart.ashx",
                            data: "sid=" + sid,
                            success: function (obj) {
                                changebgShoppingCart(obj);
                            }
                        });
                    },
                    focus: true
                },
                {
                    name: '取消',
                    callback: function () {
                    }
                }
            ]
        });
        return false;
    });

    function changebgShoppingCart(data) {
        var obj;
        try {
            obj = JSON.parse(data);
        } catch (e) {
            obj = data;
        }
        if (obj.status == 2002) {
            art.dialog.tips("很抱歉，商品数量修改失败，商品库存不足！", 1.5);
        } else {
            //构造对象
            $(".HpCartList").children().remove();
            var html = baidu.template('sc-shoppingCart', obj);
            $("#CartCheckNum").text(obj.data.sumAmount);
            //渲染
            $(".HpCartList").append(html);
            var htm = '';
            $(".FCartMTotal").html('合计:￥' + obj.data.sumTotal + '');
        }
    }

    //  活动报名
    if ($("#formActivity").length) {
        $("#formActivity").initValidform({
            ajaxPost: true,
            ignoreHidden: false,//不验证隐藏的元素
            callback: function (data) {
                if (data.status == 1001) {
                    $(".mask_div").hide();
                    //showerror(data.msg);
                    art.dialog.tips(data.msg, 1.5);
                } else if (data.status == 1000) {
                    $(".apply_info_show .name").text(data.activity.Name);
                    $(".apply_info_show .phone").text(data.activity.Mobile);
                    $(".apply_info_show .meet-id").text(data.activity.Number);

                    $(".apply_info_show").show();
                    $("#formActivity")[0].reset();
                    $("#formActivity dl").find("input").each(function () {
                        $(this).val("");
                    });
                    $("#hdId").val('${activity.id}');
                    return false;
                }
            }
        });
    }
    //活动报名
    $(".apply_info_show").click(function () {
        $(".apply_info_show").hide();
        $(".mask_div").hide();
    });
});

/**
 * 订单支付成功后处理
 * @param obj {object} 支付返回的数据
 * @param orderNo {string} 订单号
 * @param delay {number} 延迟时间
 */
function orderPaySuccessCB(obj, orderNo, delay) {
    switch (obj.status) {
        case 1000:
            art.dialog.tips("恭喜您，支付成功！", delay);
            $.ajax({
                type: "POST",
                url: "/ashx/cn/async_send_message.ashx",
                data: "orderNo=" + orderNo + "&sendType=2",
                global: false,
                async: true,//jQuery API里面所有Demo都使用false,但是这里必须使用Default的true!
                success: function () {
                    orderRedirectByOrderNo(orderNo);
                }
            });
            orderRedirectByOrderNo(orderNo, delay);
            break;
        case 1001:
            art.dialog.tips("您的余额不足以支付订单金额！<a href=\"/mobi/cn/member/pay/online.html\" style=\"color:Red; font-weight:bold;\">立即充值</a>", delay);
            break;
        case 1002:
            //art.dialog.tips("你的订单已经支付，请不要重复支付！", delay);
            orderRedirectByOrderNo(orderNo, 0.1);
            break;
        case 1006:
            art.dialog.tips("支付超时,订单已取消!", 1.5);
            setTimeout("window.location.reload()", 2000);
            break;
        default:
            art.dialog.tips("未知错误！", delay);
            break;
    }
}

/**
 * 根据订单号进行跳转
 * @param orderNo {string} 订单号
 * @param delay {number} 延迟时间
 */
function orderRedirectByOrderNo(orderNo, delay) {
    if (delay) {
        setTimeout(function () {
            orderRedirectByOrderNo(orderNo);
        }, delay * 1000);
    } else {
        switch (true) {
            case orderNo.match(new RegExp("^MS_")) !== null:
                location.href = "/mobi/cn/member/seckill/order/" + orderNo + ".html";
                break;
            case orderNo.match(new RegExp("^GB_")) !== null:
                location.href = "/mobi/cn/member/bulk/order/" + orderNo + "_info.html";
                break;
            case orderNo.match(new RegExp("^BG_")) !== null:
                location.href = "/mobi/cn/member/buy_gang/order/" + orderNo + "_info.html";
                break;
            case orderNo.match(new RegExp("^PM_")) !== null:
                location.href = "/mobi/cn/pay_success.html?orderNo=" + orderNo;
                break;
            default:
                art.dialog.tips("重定向错误！", delay);
                break;
        }
    }
}

/**
 * 查看门店地图
 * @param salesOutletsId {number} 门店Id
 */
function salesOutletsLocation(salesOutletsId) {
    if (salesOutletsId.length <= 0) {
        art.dialog.tips("请先选择自提门店！", 2);
        return false;
    }
    if (salesOutletsId == 0) {
        art.dialog.tips("请先选择自提门店！", 2);
        return false;
    }
    window.open('/mobi/cn/salesoutlets_mapshow/' + salesOutletsId + '.html')
    return false;
}