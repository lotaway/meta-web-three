// import crypto from '../utils/crypto';
import Base from './base';

class Cart extends Base {

    constructor(...params) {
        super(...params);
    }

    /**
     * 获取当前用户的购物车信息（通过session）
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 失败回调
     */
    getShoppingCart(successCB, errorCB = null) {
        this.signRequest("GetShoppingCart", "", successCB, errorCB);
    }

    add(...params) {
        this.handler("add", ...params);
    }

    /**
     * 清空购物车
     */
    deleteAll(...params) {
        this.handler("sid=all", ...params);
    }

    /**
     * 删除多个商品
     * @param goodsArray {Array} 商品数组，
     * @param params
     */
    deleteSome(goodsArray, ...params) {
        var sData = ""  //  商品请求数据
            , gData = ""    //  套餐请求数据
        ;
        goodsArray.forEach(function (item) {
            if (item["sid"]) {
                sData += `,${item["sid"]}`;
            }
            else if (item["gpid"]) {
                gData += `,${item["gpid"]}`;
            }
            else {
                throw new Error("deleteSome缺少商品/套餐标识，无法删除");
            }
        });

        sData.replace(/^,/, "sid=");
        if (gData !== "") {
            gData.replace(/^,/, (sData !== "" ? '&' : '') + "gpid=");
        }

        this.handler(sData + gData, ...params);

    }

    //  删除单个商品
    deleteOne(cid, isPackage = false, ...params) {
        this.handler(`${isPackage ? "gpid" : "sid"}=${cid}`, ...params);
    }

    //  增加商品数量
    plusNum(cid, oldAmount, ...params) {
        this.changeNum(cid, ++oldAmount, ...params);
    }

    //  减少商品数量
    minusNum(cid, oldAmount, ...params) {
        this.changeNum(cid, --oldAmount, ...params);
    }

    //  修改商品数量
    changeNum(cid, amount, ...params) {
        if (amount != 0) this.handler(`cid=${cid}&amount=${amount}`, ...params);
    }

    /**
     * 操作购物车
     * @param data {String} 操作参数
     * @param successCB {Function} 成功回调
     * @param errorCB {Function} 失败回调
     */
    handler(data, successCB, errorCB = null) {
        $("#edit-shoppingcart").delegate(".amountJia", "click", function () {
            //商品数量加一
            var obj = $(this).attr("data-sid")
                , data = "cid=" + obj + "&amount=" + (parseInt($("#amount" + obj).val()) + 1)
            ;
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
            }
            else {
                let data = "cid=" + obj + "&amount=" + $("#amount" + obj).val();
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
            var obj = $(this).attr("data-geid"),
                tn = 0;
            $('.' + obj + '').toggleClass('cur');
            $('.CartProChecked.cur').each(function () {
                tn += Number($(this).parents(".CartLi").find(".CartZPrice i").text() || $(this).parents(".CartLi").find(".CartZPrice i").text()) * $(this).parents(".CartLi").find(".changeAmount").val() || 0;
            });
            $("#CartTotalNum").text(tn.toFixed(2));
            if (!$('.' + obj + '').hasClass('cur')) {
                $('.CartProAllChecked').removeClass('cur');
            }
        });
        //  购物车，增删加减改都调这个
        function changeShoppingCart(data, bidAppend) {
            $.ajax({
                type: "POST",
                url: "/ashx/cn/shopping_cart.ashx",
                data: data,
                //  清空购物车 data='sid=all'
                dataType: "json",
                success: function (obj) {
                    switch (obj.status) {
                        case 1001:
                            art.dialog.tips('您购买的商品总额最多换购' + obj.amount + '件此商品！', 1.5);
                            break;
                        case 1002:
                            art.dialog.tips("很抱歉，换购失败，商品库存不足！", 1.5);
                            break;
                        case 1003:
                            art.dialog.tips("已达到最大数量", 1.5);
                            break;
                        case 2000:  //  2000是增减改商品数量成功
                        case 3000:  //  3000是删除商品成功
                            //  成功，用百度模板重新显示购物车内的商品信息
                            if (!bidAppend) {
                                $(".CartList").children().remove();
                                var html = baidu.template('sc-shoppingCart', obj);
                                $("#aSettlement").html('去结算(' + obj.data.sumAmount + ')');
                                $(".CartList").append(html);
                                $("#CartTotalNum").html(obj.data.sumTotal);
                                if (typeof appCartChangeHandler == 'function') appCartChangeHandler(obj.data.sumAmount);
                                //$('.CartProAllChecked,.CartProChecked').removeClass('cur');
                            }
                            break;
                        case 2001:
                            art.dialog.tips("很抱歉，此操作导致商品总额不足换购商品，商品数量修改失败！", 1.5);
                            break;
                        case 2002:
                            art.dialog.tips("很抱歉，商品数量修改失败，商品库存不足！", 1.5);
                            break;
                        case 3001:
                            art.dialog.tips("很抱歉，此操作导致商品总额不足换购商品，商品删除失败！", 1.5);
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
            if (!$('.CartProChecked').hasClass('cur')) {
                art.dialog.tips("请选择商品！", 1.5);
                return false;
            }
            $.ajax({
                type: "POST",
                url: "/ashx/cn/detection_session.ashx",
                success: function (obj) {
                    switch (obj.status) {
                        case 1000:
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
                                }
                                else {
                                    Datageid = "_" + $(this).attr('data-geid') + "";
                                }
                                geid = geid + Datageid;
                            });
                            window.location = "/mobi/cn/order/submit/" + geid + "" + Pgeid + ".html";
                            break;
                        case 1001:
                            //登录注册弹窗 显示
                            window.location = "/mobi/cn/login.html";
                            break;
                        case 1002:
                            art.dialog.tips("购物车目前没有加入任何商品！", 1.5);
                            break;
                        default:
                            break;
                    }
                }
            });
            return false;
        });
    }

}

/**
 * 用户相关操作接口
 * http://ask.dcloud.net.cn/article/157 登录令牌
 */
class User extends Base {

    constructor() {
        super();
        this.cart = new Cart();
    }

    /**
     * 获取当前用户的购物车信息
     */
    getShoppingCart(...params) {
        this.cart.getShoppingCart(...params);
    }

    /**
     * 传输客户端标识，用于推送
     * @param params {Object} 请求参数
     * @param successCB {Function} 成功回调
     * @param errorCB {Function} 失败回调
     */
    judgeAppId(params = {}, successCB, errorCB = null) {
        const defaultParams = {
            username: "",   //  用户名
            clientAppId: "" //  客户端推送标识
        };
        const finalParams = this.paramHandler(defaultParams, params);
        this.signRequest("JudgeAppId", "", statusCode => {
            var data = {
                msg: "" //  信息
                , statusCode: statusCode
            };
            switch (data.statusCode) {
                //  包含的用户名已绑定过（会重新进行绑定）
                case 1000:
                    data.msg = "包含的用户名已绑定过";
                    break;
                //  未包含用户名，但客户端标识已绑定用户
                case 1001:
                    data.msg = "未包含用户名，但客户端标识已绑定用户";
                    break;
                //  未包含用户名，且客户端标识未绑定用户，但已存在于匿名用户表里，只更新最后活跃时间
                case 1002:
                    data.msg = "已存在匿名用户表里";
                    break;
                //  未包含用户名，且客户端标识未绑定用户、也未存在于匿名用户表里，进行存储到匿名用户表里
                case 1003:
                    data.msg = "未登记过，正常登记";
                    break;
                //  异常情况
                case 1004:
                    data.msg = "推送登记失败";
                    this.errorHandler(data.msg);
                    break;
            }
            successCB(data);
        }, errorCB, finalParams);
    }

    /**
     * 检测是否有session可以进行部分用户操作（如加入购物车）
     */
    detection_session() {
// /ashx/cn/detection_session.ashx
    }


    /**
     * 注册用户
     * @param {Object} params 参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 失败回调
     */
    addUser(params = {}, successCB, errorCB = null) {
        //  todo 注册用户， 默认参数、接口名称、返回数据
        const defaultParams = [
            ["mobile"],
            ["pwd"],
            ["username"]
        ];
        const finalParams = this.paramHandler(defaultParams, params);
        const noDecryptParams = "isDecrypt=" + params.isDecrypt != undefined ? params.isDecrypt : "false"; //    是否加密传输
        this.signRequest('AddUser', finalParams, successCB, errorCB, noDecryptParams);

    }

    /**
     * 比对用户名和密码，返回用户信息
     * @param {Object} params 参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 失败回调
     */
    getUserByUserName(params = {}, successCB, errorCB) {
        // var desKey = App.getConfig('desKey');
        const defaultParams = [
                ["bandType"],   //  登录类型，跟第三方登录有关？
                ["nickname"],   //  昵称
                ["openid"],     //  第三方登录openid
                ["pwd"],        //  密码
                ["username"]   //  用户名
            ]
            , saveDayKey = "saveDay"
            , defaultNoSignedParams = {
                saveDay: 30    //  保持登录的天数，默认为0
            }
            , finalParams = this.paramHandler(defaultParams, params)
            ,
            noDecryptParams = "isDecrypt=" + (params.isDecrypt != undefined ? params.isDecrypt : "false") + "&clientAppId=" + params.clientAppId //    不加密参数，isDecrypt表示不加密用户名和密码
            , saveDayValue = params[saveDayKey] || defaultNoSignedParams[saveDayKey];

        //  本地加密
        /*if (data.isDecrypt) {
         if (data.username) data.username = crypto.NativeDes.strToHex(crypto.NativeDes.encrypt(desKey, data.username, desKey));
         if (data.pwd) data.pwd = crypto.NativeDes.strToHex(crypto.NativeDes.encrypt(desKey, data.pwd, desKey));
         if (data.openid) data.openid = crypto.NativeDes.strToHex(crypto.NativeDes.encrypt(desKey, data.openid, desKey));
         }*/

        this.signRequest('GetUserByUserName', finalParams, result=> {

            if (result !== 1001 && result !== 1002 && result !== 1003) {
                // var desKeyUser = App.getConfig('desKeyUser');
                // let value = crypto.NativeDes.encrypt(desKeyUser, data.username, desKeyUser);

                // alert(JSON.stringify(result));

                // 先保存登录状态
                this.App.storage('username', params.username);
                this.App.storage('pwd', params.pwd);
                this.App.cookie(this.App.getConfig('userCookieName'), result.md5EncryptUserName, saveDayValue * 24 * 60 * 60 * 1000);
            }
            //  再执行成功回调
            successCB(result);
        }, errorCB, noDecryptParams);
    }

    /**
     * 根据用户名获取订单信息
     * @param {Object} params 请求参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 失败回调
     */
    getOrderListByUser(params = {}, successCB, errorCB) {
        const defaultData = {
            bandType: ""
        };

        for (let p in defaultData) {
            if (params[p] == undefined) {
                params[p] = defaultData[p];
            }
        }
        //  todo username需要加密，然而des加密和csharp采用的方式不一致
        this.signRequest('GetOrderListByUser',
            `num=${params.num}&order=${params.order}&orderValue=${params.orderValue}&page${params.page}&username=${params.username}&type=${params.datatype}&typeValue=${params.typeValue}`, successCB, errorCB);
    }
}

export default User;