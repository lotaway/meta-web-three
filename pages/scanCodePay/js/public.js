$(function () {

    jQuery.ajaxSetup({dataType: "json"});

    //  公告关闭事件
    var home_tx = "home_tx";
    if (document.getElementById(home_tx)) {
        $("#btn_home_tx_close").click(function () {
            $("#" + home_tx).remove();
        });
    }

    //  在app环境下因没有第三方登录功能，暂时隐藏第三方关联信息，待制作完再去除
    if (document.body.hasClass("member-msg") && MainController.application.check() && document.getElementById("oauth_app_list")) {
        document.getElementById("oauth_app_list").addClass("hide");
    }

    //  微信内置浏览器环境
    if (MainController.inWechat()) {
        // 支付方式-支付宝在微信内不可用，目前是提示复制网址到其他浏览器即可进行支付，无须隐藏
        /*$("input[name=pay_bank]").each(function () {
         if ($(this).val() == 'directPay') {
         $(this).parents('.item_pay_bank').remove();
         }
         });*/
        //  在微信里隐藏跟微信相关的第三方登录
        if ($("body").hasClass("member-msg")) {
            $(".account-link li").each(function () {
                if ($(this).find("span").text().indexOf("微信") > -1) {
                    $(this).hide();
                }
            });
        }
        if ($("body").hasClass("page-login")) {
            $(".ContainerLoginOther .LoginOther a").each(function () {
                if ($(this).find(".name").text().indexOf("微信") > -1) {
                    $(this).hide();
                }
            });
        }
        //  微信小程序环境
        MainController.weChatMiniProgram.checks()
            .fail(MainController.wechatShareInit);
    }
    //  微信小程序环境
    MainController.weChatMiniProgram.checks()
        .done(MainController.weChatMiniProgram.init);
    //  2016版app环境判断
    if (MainController.application.check()) {
        MainController.application.init();
    }
    //  APP相关处理
    appHandler(false);

    //  指定的链接添加返回的页面
    $("a.btn-add-href").click(function () {
        var value = getUrlParam('redirect');
        $(this).attr('href', $(this).attr('href') + '?redirect=' + encodeURI(value != null ? value : location.href));
    });

    //  返现订单页 计算当前返现百分比
    $(".user-cash-back-list .data-process").each(function () {
        var t = $(this);
        var now = (100 * t.find('.cashBackTotal').data("now").toString().split("元")[0] / t.find('.cashBackTotal').text().toString().split("元")[0]).toFixed(1) + '%';
        t.find(".rate").css("width", now);
        t.find(".board").text(now);
    });

    //  会员中心首页才显示顶部的个人信息与快捷入口
    if ($('body').hasClass('user-index')) {
        if (/member\/\d+$/.test(window.location.href)) {
            $(".MenberH,.MenLists:first").addClass('hide');
            //  赋予标题
            if (/member\/$/.test(document.referrer) && localStorage.getItem('pageTitle')) {
                $(".HeadM").html(localStorage.getItem('pageTitle'));
            }
        }

        var saveTitle = function (target, titleTarget) {
            target.click(function () {
                if (!/member\/\d+$/.test($(this).attr('href'))) return true;
                localStorage.setItem('pageTitle', (titleTarget ? $(this).find(titleTarget) : $(this)).text().replace(/\([^\(\)]*\)/, ""));
            });
        };
        saveTitle($(".MenLists a"), ".MenList");
        saveTitle($(".MenuList a"));
    }

    //  订单详情页
    $(".OrderInfoH span").addClass("cur");

    //回到顶部
    if ($('.goTop').length) {
        scrollListener(function ($win) {
            $win > 100 ? $('.goTop').removeClass('transparent') : $('.goTop').addClass('transparent');
        });
        $('.goTop').hammer().on('tap', function (e) {
            e.gesture.srcEvent.preventDefault();
            $('html,body').animate({scrollTop: 0}, 300);
        });
    }

    //头部下拉菜单
    if ($('.menu').length) {
        $('.menu').hammer().on('tap', function () {
            $('.MenuList').slideToggle(100);
        });
    }
    $('main').click(function () {
        $('.MenuList').slideUp(100);
    });
    if ($('footer').length) {
        $('footer').hammer().on('tap', function () {
            $('.MenuList').slideUp(100);
        });
    }

    //频道页分类下拉列表
    if ($('.ChNavListH').length) {
        function ChNavHUp() {
            $('.ChNavListH').slideUp(100);
        }

        $('.ChNavmore').hammer().on('tap', function () {
            $('.ChNavListH').slideDown(100);
        });
        $('.ChNav').nextAll().click(ChNavHUp);
        $('.ChNavUp').hammer().on('tap', ChNavHUp);
        $('header').hammer().on('tap', ChNavHUp);
        $('footer').hammer().on('tap', ChNavHUp);
    }

    //PayingType
    $('.PayType input[type=radio]').each(function () {
        if ($(this).is(':checked')) {
            payWayChangeHandler.call(this);
        }
        $(this).hammer().on('tap', function () {
            if (!$(this).is(':checked')) {
                payWayChangeHandler.call(this);
            }
        })
    });
    if ($('.PayType input[type=radio]:checked').length === 0) $('.PayType input[type=radio]').eq(0).prop('checked', true).trigger('click');
    //积分抵扣时的实付总额
    $('#cbIntegral').click(function () {
        if ($(this).is(':checked')) {
            $('#actualPayAll').hide();
            $('#actualPayIntegral').show();
        } else {
            $('#actualPayAll').show();
            $('#actualPayIntegral').hide();
        }
    });

    $('.CheckCode').click(function () {
        $('.PopBox_Code,.shadow').show();
        $('body').addClass('hidden');
    });
    $('.shadow,.PopBox_Code .btn-close').click(function () {
        $('body').removeClass('hidden');
        $('.PopBox_Code,.shadow').hide();
    });
    document.addEventListener("touchmove", function (e) {
        if ($("body").hasClass("hidden")) {
            e.preventDefault();
        }
    });

    //prshow tab

    if ($('.PsDiscount').length) {
        $('.PsDiscount').hammer().on('tap', function () {
            $('.PsDiscountr').toggleClass('PsDiscountrH');
            $('.PsDisArrow').toggleClass('PsDisArrowH');
        });
    }

    MainController.page.goods();

    /**
     * 转换值
     * @param tranValue {String} 需要转换的值
     * @return {*}
     * @constructor
     */
    function NumTxtTranValue(tranValue) {
        if (tranValue == "" || tranValue == null) {
            tranValue = "0";
        }
        return tranValue;
    }

    //  输入数量事件
    $(".NumTxt").keyup(function () {
        var tranValue = NumTxtTranValue($(this).val());
        if (!$(this).hasClass("Crowdfund") && !$(this).hasClass("prepro")) {
            var MaxVal = parseInt($(this).siblings(".Maximum").text());
            var MaxValLen = $(this).siblings(".Maximum").html().length;
            if (parseInt(tranValue) > MaxVal) {
                tranValue = MaxVal;
            } else {
                tranValue = tranValue.replace(/[^\d]/g, '');
                if (tranValue.length > 1 && tranValue[0] === "0") {
                    tranValue = NumTxtTranValue(tranValue.replace(/^0+/, ''));
                }
                if (tranValue.length > MaxValLen) {
                    tranValue = tranValue.substring(0, MaxValLen);
                }
            }
            $(this).val(tranValue);
        }
    });
    //全选
    $("input#checkbox").click(function () {
        if (this.checked) {
            $("input[type='checkbox']").each(function () {
                this.checked = true;
            });
        } else {
            $("input[type='checkbox']").each(function () {
                this.checked = false;
            });
        }
    });
    //  订单提交中的收货类型切换事件
    $('.OrderHList li').each(function (index) {
        if ($(this).find('input:checked').val() == "0") {
            $('.OrderExpress').show();
            $('.OrderHList li').eq(0).addClass('OrderNavH').siblings('li').removeClass('OrderNavH');
            $('.OrderSendMode').eq(0).show().siblings('.OrderSendMode').hide();
        } else if ($(this).find('input:checked').val() == "1") {
            $('.OrderExpress').hide();
            $('.OrderHList li').eq(1).addClass('OrderNavH').siblings('li').removeClass('OrderNavH');
            $('.OrderSendMode').eq(1).show().siblings('.OrderSendMode').hide();
        }
        $(this).hammer().on('tap', function () {
            var freight
                , sSumFreightParent = $("#sSumFreight").parents("dd")
            ;
            if ($(this).hasClass("OrderNavH")) return false;
            $(this).addClass('OrderNavH').siblings('li').removeClass('OrderNavH');
            $('.OrderSendMode').eq(index).show().siblings('.OrderSendMode').hide();
            if ($('#PickedBt').is(':checked')) {
                $('.OrderExpress').show();
                sSumFreightParent.show();
                //  显示配送时间
                $("#OrderTime").show();
            }
            //  自提时
            else {
                $('.OrderExpress').hide();
                sSumFreightParent.hide();
                $("#OrderTime").hide();
                freight = 0;
            }
            checkDeliveryWay(freight);
        });
    });

    //商品收藏
    $('.CollectCheck').each(function () {
        $(this).hammer().on('tap', function (e) {
            e.gesture.srcEvent.preventDefault();
            $(this).toggleClass('cur');
            if (!$(this).hasClass('cur')) {
                $('.CollectAllDelCheck').removeClass('cur');
            }
        })
    });

    if ($('.CollectAllDelCheck').length) {
        $('.CollectAllDelCheck').hammer().on('tap', function () {
            if ($(this).hasClass('cur')) {
                $(this).removeClass('cur');
                $('.CollectCheck').each(function () {
                    $(this).removeClass('cur');
                });
            } else {
                $(this).addClass('cur');
                $('.CollectCheck').each(function () {
                    $(this).addClass('cur');
                });
            }

        });
    }


    if ($('.CollectOpenDelBt').length) {
        $('.CollectOpenDelBt').hammer().on('tap', function () {
            $(this).addClass('cur').siblings('a').removeClass('cur');
            $('.CollectCheck').addClass('del');
            $('.CollectFoot').show().siblings('ul').hide();
        });
    }
    if ($('.CollectDelCancelBt').length) {
        $('.CollectDelCancelBt').hammer().on('tap', function () {
            $(this).addClass('cur').siblings('a').removeClass('cur');
            $('.CollectCheck').removeClass('del').removeClass('cur');
            $('.CollectAllDelCheck').removeClass('cur');
            $('.CollectFoot').hide().siblings('ul').show();
        });
    }

    if ($('.CartProAllChecked').length) {
        $('.CartProAllChecked').hammer().on('tap', function () {
            var tn = 0
                , activeClass = "cur"
            ;

            if ($(this).hasClass(activeClass)) {
                $(this).removeClass(activeClass);
                $('.CartProChecked').removeClass(activeClass);
            } else {
                $(this).addClass(activeClass);
                $('.CartProChecked').each(function (index, item) {
                    if (!$(item).hasClass("disabled")) {
                        $(item).addClass(activeClass).attr('checked', 'checked');
                    }
                });
            }
            cartCheckChange();
        });
    }

    //  结算中心的收货地址选择
    $('.OrderAddressList dd').each(function () {
        if ($(this).find('input[name=userAddress]').is(':checked')) {
            $(this).addClass('DefaultAddr').siblings('dd').removeClass('DefaultAddr');
            var htm = '<a href="javascript:void(0);"><div class="Consignee clearf"><div class="floatl">' + $(this).find('.floatl').text() + '</div><div class="floatr">' + $(this).find('.floatr').text() + '</div></div><div class="AddAddrDet">' + $(this).find('.AddAddrDet').text() + '</div></a>';
            $('.AddAddr').html(htm);
        }

        $(this).hammer().on('tap', function () {
            $(this).addClass('DefaultAddr').siblings('dd').removeClass('DefaultAddr');
            var htm = '<a href="javascript:void(0);"><div class="Consignee clearf"><div class="floatl">' + $(this).find('.floatl').text() + '</div><div class="floatr">' + $(this).find('.floatr').text() + '</div></div><div class="AddAddrDet">' + $(this).find('.AddAddrDet').text() + '</div></a>';
            $('.AddAddr').html(htm);
            toggleNav(false);
        });
    });

    //   服务协议，点击关闭
    $("#btnAgreeServicet").click(function () {
        window.history.back();
    });

    //  销售门店 切换tab
    $('.MapIntroH li').each(function (index) {
        $(this).click(function () {
            $(this).addClass('cur').siblings().removeClass('cur');
            $('.MapIntroCon').eq(index).show().siblings().hide();
        })
    });
});

$.fn.dxCommentShowImg = function (config) {
    config = config || {};
    var _config = {
        flickContainerLink: config.flickContainerLink || '.flicking_con a'
        , evalShowImgThumb: config.evalShowImgThumb || '.EvalShowImgThumb'
        , evalShowImgThumbItem: config.evalShowImgThumbItem || 'dd'
        , showImgClose: config.showImgClose || '.ShowImgClose'
        , mainVisual: config.mainVisual || '.main_visual'
        , cdMainContent: config.cdMainContent || '.cd-main-content'
        , mainZIndexClassName: config.mainZIndexClassName || 'mainZIndex'
        , mainImage: config.mainImage || '.main_image'
        , onClassName: config.onClassName || 'on'   //  当前所在位置样式名称
    };

    $(this).each(function () {
        var $self = $(this)
            , $ThisNav = $self.find(_config.flickContainerLink);
        $self.find(_config.evalShowImgThumb + ' ' + _config.evalShowImgThumbItem).click(function (e) {
            var $PageIndex = $self.index() + 1
                , nodeMainVisual = $self.find(_config.evalShowImgThumb).siblings(_config.mainVisual)
            ;
            $(_config.cdMainContent).addClass(_config.mainZIndexClassName);
            nodeMainVisual.show().find(_config.mainImage).touchSlider({
                flexible: true,
                speed: 200,
                paging: $ThisNav,
                page: $PageIndex,
                counter: function (e) {
                    $ThisNav.removeClass(_config.onClassName).eq(e.current - 1).addClass(_config.onClassName);
                }
            });
        });

        $self.find(_config.showImgClose).click(function () {
            $(_config.cdMainContent).removeClass(_config.mainZIndexClassName);
            $(_config.mainVisual).hide();
        });
    });
    return this;
};

/**
 * 商品图文详情缩放查看大图
 * @param config {object} 配置项
 * @returns {object}
 */
$.fn.goodsDetailImageShow = function (config) {
    config = config || {};
    var $self = $(this)
        , _config = {
            containerId: config.containerId || "containerMainVisual"    //  容器名称
            , flickContainer: config.flickContainer || 'flicking_con'
            , flickContainerLink: config.flickContainerLink || 'a'
            , showImgClose: config.showImgClose || 'ShowImgClose'
            , mainVisual: config.mainVisual || 'main_visual'
            , cdMainContent: config.cdMainContent || '.cd-main-content'
            , mainZIndexClassName: config.mainZIndexClassName || 'mainZIndex'
            , mainImage: config.mainImage || 'main_image'
            , onClassName: config.onClassName || 'on'   //  当前所在位置样式名称
            , callback: config.callback || null //  完成回调
        }
        , mainVisual = ''   //  指示点
        , mainImage = ''    //  图片滑动容器
        , container    //  容器
        , touchSliderEventAdded = [] //  是否已添加滑动事件状态
        , windowScrollTop = 0   //  记录滚动位置
        , $ThisNav
    ;
    //  生成大图展示内容
    $self.each(function (index, item) {
        mainVisual += '<a href="javascript:void(0);"></a>';
        mainImage += '<li><div class="pinch-zoom"><img class="" src="' + item.getAttribute(item.hasClass("lazy") ? "data-original" : "src") + '" /></div></li>';
        //  图片点击进入大图展示
        $(this).click(function (e) {
            var $self = $(this)
            ;
            windowScrollTop = document.body.scrollTop;
            $(_config.cdMainContent).addClass(_config.mainZIndexClassName);
            container.show();
            //  添加滑动事件
            if (touchSliderEventAdded !== true) {
                container.find("." + _config.mainImage).touchSlider({
                    flexible: true,
                    speed: 200,
                    paging: $ThisNav,
                    page: 1,
                    counter: function (e) {
                        $ThisNav.removeClass(_config.onClassName).eq(e.current - 1).addClass(_config.onClassName);
                    }
                });
                touchSliderEventAdded = true;
            }
            container.find("." + _config.flickContainer + " " + _config.flickContainerLink).eq(index).trigger("click");
        });
    });

    $("body").append('<div id="' + _config.containerId + '" class="' + _config.mainVisual + '"><div class="' + _config.flickContainer + '">' + mainVisual + '</div><div class="' + _config.mainImage + '"><ul>' + mainImage + '</ul></div><div class="' + _config.showImgClose + '"></div></div>');
    container = $("#" + _config.containerId);
    $ThisNav = container.find("." + _config.flickContainer + " " + _config.flickContainerLink);
    container.find("." + _config.showImgClose).click(function () {
        window.scrollTo(0, windowScrollTop);
        $(_config.cdMainContent).removeClass(_config.mainZIndexClassName);
        $("." + _config.mainVisual).hide();
    });
    if (typeof _config.callback === "function") {
        _config.callback(this);
    }

    return this;
};

/**
 * 窗口高宽变化事件监听
 * @function resizeListener
 * @param callback{function} 回调
 * @return w{object} 窗口对象的宽高
 */
var resizeListener = (function ($) {
    var timer, fnArray = [];
    $(window).resize(function () {
        if (timer) clearTimeout(timer);
        timer = setTimeout(function () {
            var w = {
                width: $(window).width(),
                height: $(window).height()
            };
            fnArray.forEach(function (item) {
                item(w);
            });
        }, 16);
    });
    $(window).trigger("resize");
    return function (callback) {
        if (typeof callback === 'function') fnArray.push(callback);
    };
})($ || {});
/**
 * 窗口滚动事件监听
 * @function scrollListener
 * @param callback{function} 回调
 * @return scrollTop 窗口滚动的距离
 */
var scrollListener = (function ($) {
    var timer, scrollFn = [];
    $(window).scroll(function () {
        if (timer) clearTimeout(timer);
        timer = setTimeout(function () {
            var scrollTop = $(window).scrollTop();
            scrollFn.forEach(function (item) {
                item(scrollTop);
            });
        }, 16);
    });
    $(window).trigger("scroll");
    return function (callback) {
        if (typeof callback === 'function') scrollFn.push(callback);
    };
})($ || {});

/**
 * 在旧app隐藏底部切换导航
 * @param value {Boolean} 是否直接隐藏导航栏、选项卡
 */
function appHandler(value) {
    var w = window.navigator.userAgent;
    //  旧APP兼容判断
    if (w.indexOf("micronetapp") > -1) {
        $('.tab-footer').addClass('hide');
    }
}

//  版权解码
function decode(s) {
    return s.replace(/.{4}/g, function ($) {
        return String.fromCharCode(parseInt($, 16));
    });
}

//  投票弹窗
function showPopMethod() {
    var result = $(".result");
    if (result) {
        $(".btn-close-window").trigger("click");
        result.parent().show();
        result.show();
        document.body.addClass("hidden");
        setTimeout(function () {
            result.removeClass("hide");
            result.parent().removeClass("hide");
        }, 0);
    }
}

/**
 * 获取请求参数
 * @param name {String} 参数名
 * @return {String|null} 参数值
 * */
function getUrlParam(name) {
    if (!name) return null;
    //构造一个含有目标参数的正则表达式对象匹配目标参数
    var r = window.location.search.substr(1).match(new RegExp("(^|&)" + name + "=([^&]*)(&|$)"));
    return r == null ? null : decodeURI(r[2]);
}

/**
 * 设置请求参数
 * @param {Object} newParams 新参数
 * @param {String} url 链接
 * @returns {String|Boolean} 返回的地址
 */
function setUrlParam(newParams, url) {
    var finalParams = "";

    url = url || location.href;
    if (!newParams || typeof newParams !== "object") return false;
    if (url.indexOf("?") > -1) {
        url.split("?")[1].split('&').forEach(function (param) {
            var paramArray = param.split("=")
                , key = paramArray[0]
                , newValue = newParams[key]
            ;

            if (newValue === undefined) {
                finalParams += "&" + param;
            } else if (newValue !== null && newValue !== "") {
                finalParams += "&" + key + "=" + newValue;
            }
            delete newParams[key];
        });
    }
    for (var param in newParams) {
        if (newParams.hasOwnProperty(param)) {
            finalParams += "&" + param + "=" + newParams[param];
        }
    }
    return url.split('?')[0] + finalParams.replace(/^&/, "?");
}

/**
 * 对Date的扩展，将 Date 转化为指定格式的String
 * 月(M)、日(d)、12小时(h)、24小时(H)、分(m)、秒(s)、周(E)、季度(q) 可以用 1-2 个占位符
 * 年(y)可以用 1-4 个占位符，毫秒(S)只能用 1 个占位符(是 1-3 位的数字)
 * eg:
 * (new Date()).pattern("yyyy-MM-dd hh:mm:ss.S") ==> 2006-07-02 08:09:04.423
 * (new Date()).pattern("yyyy-MM-dd E HH:mm:ss") ==> 2009-03-10 二 20:09:04
 * (new Date()).pattern("yyyy-MM-dd EE hh:mm:ss") ==> 2009-03-10 周二 08:09:04
 * (new Date()).pattern("yyyy-MM-dd EEE hh:mm:ss") ==> 2009-03-10 星期二 08:09:04
 * (new Date()).pattern("yyyy-M-d h:m:s.S") ==> 2006-7-2 8:9:4.18
 */
Date.prototype.pattern = function (fmt) {
    var o = {
        "M+": this.getMonth() + 1, //月份
        "d+": this.getDate(), //日
        "h+": this.getHours() % 12 == 0 ? 12 : this.getHours() % 12, //小时
        "H+": this.getHours(), //小时
        "m+": this.getMinutes(), //分
        "s+": this.getSeconds(), //秒
        "q+": Math.floor((this.getMonth() + 3) / 3), //季度
        "S": this.getMilliseconds() //毫秒
    };
    var week = {
        "0": "/u65e5",
        "1": "/u4e00",
        "2": "/u4e8c",
        "3": "/u4e09",
        "4": "/u56db",
        "5": "/u4e94",
        "6": "/u516d"
    };
    if (/(y+)/.test(fmt)) {
        fmt = fmt.replace(RegExp.$1, (this.getFullYear() + "").substr(4 - RegExp.$1.length));
    }
    if (/(E+)/.test(fmt)) {
        fmt = fmt.replace(RegExp.$1, ((RegExp.$1.length > 1) ? (RegExp.$1.length > 2 ? "/u661f/u671f" : "/u5468") : "") + week[this.getDay() + ""]);
    }
    for (var k in o) {
        if (new RegExp("(" + k + ")").test(fmt)) {
            fmt = fmt.replace(RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (("00" + o[k]).substr(("" + o[k]).length)));
        }
    }
    return fmt;
};

////////////////////////格式化日期
Date.prototype.Format = function (formatStr) {
    var str = formatStr;
    var Week = ['日', '一', '二', '三', '四', '五', '六'];
    str = str.replace(/yyyy|YYYY/, this.getFullYear());
    str = str.replace(/yy|YY/, (this.getYear() % 100) > 9 ? (this.getYear() % 100).toString() : '0' + (this.getYear() % 100));
    str = str.replace(/MM/, this.getMonth() + 1 > 9 ? (this.getMonth() + 1).toString() : '0' + (this.getMonth() + 1));
    str = str.replace(/M/g, this.getMonth() + 1);
    str = str.replace(/w|W/g, Week[this.getDay()]);
    str = str.replace(/dd|DD/, this.getDate() > 9 ? this.getDate().toString() : '0' + this.getDate());
    str = str.replace(/d|D/g, this.getDate());
    str = str.replace(/hh|HH/, this.getHours() > 9 ? this.getHours().toString() : '0' + this.getHours());
    str = str.replace(/h|H/g, this.getHours());
    str = str.replace(/mm/, this.getMinutes() > 9 ? this.getMinutes().toString() : '0' + this.getMinutes());
    str = str.replace(/m/g, this.getMinutes());
    str = str.replace(/ss|SS/, this.getSeconds() > 9 ? this.getSeconds().toString() : '0' + this.getSeconds());
    str = str.replace(/s|S/g, this.getSeconds());
    return str;
};

/**
 * String类型转换成Date类型
 * @param DateStr 字符串类型的日期
 * @returns {Date} 日期类型的日期
 */
function stringToDate(DateStr) {
    var args;
    var myDate = new Date(Date.parse(DateStr));
    if (isNaN(Number(myDate))) {
        args = DateStr.split('-');
        myDate = new Date(args[0], --args[1], args[2]);
    }
    return myDate;
}

/**
 * 错误处理，考虑？
 * @param error 错误信息
 */
function errorHandler(error) {
    console.error('错误：' + error);
}

/**
 * 检查元素是否拥有样式名
 * @param c 样式名
 * @returns {boolean}
 */
Element.prototype.hasClass = function (c) {
    return new RegExp("(^|\\s)" + c + "(\\s|$)").test(this.className);
};
/**
 * 把样式名赋给元素
 * @param c 样式名
 */
Element.prototype.addClass = function (c) {
    !this.hasClass(c) && (this.className += " " + c);
    return this;
};
/**
 * 从元素中移除样式名
 * @param c 样式名
 */
Element.prototype.removeClass = function (c) {
    this.hasClass(c) && (this.className = this.className.replace(new RegExp("(^|\\s)" + c + "(\\s|$)"), ""));
    return this;
};

/**
 * 手机发送验证码后倒计时
 * @param time 倒计时的总时间
 * @param target 倒计时显示所在的元素
 * @param text 倒计时中显示的文本 用#{time}表示动态时间
 * @param options {object} 更多参数
 */
function updateTimeLabel(time, target, text, options) {
    var handler,
        btn = target || document.getElementById("btnMobileCode");

    options = options || {};
    options.countCB = options.countCB || function () {

    };
    options.endCB = options.endCB || function () {

    };
    if (btn.length) btn = btn[0];
    text = text || '#{time}s后重新发送';
    if (!btn.text) btn.text = btn.value;
    btn.addClass("finished").value = time <= 0 ? "免费获取验证码" : text.replace('#{time}', time);
    handler = setInterval(function () {
        if (--time > 0) {
            var newText = text.replace('#{time}', time);
            btn.value = newText;
            options.countCB(time, newText);
        } else {
            clearInterval(handler);
            handler = null;
            btn.removeClass("finished").value = btn.text;
            options.endCB(btn);
        }
    }, 1000);

    return handler;
}

//  支付方式变化处理事件
function payWayChangeHandler() {
    if ($(this).val() == "预存款" || $(this).val() == "货到付款") {
        $('.PayTypeYuMoney').hide();
        $('#cbPreDeposit').removeProp('checked');
    } else {
        $('.PayTypeYuMoney').show();
    }
    $(this).parents('li').addClass('border').siblings('li').removeClass('border');

    //积分抵扣
    if ($(this).val() == "货到付款") {
        $(".OrderPayWayIntegral").hide();
        $('#cbIntegral').prop('checked', false);
        $('#actualPayAll').show();
        $('#actualPayIntegral').hide();
    } else {
        $(".OrderPayWayIntegral").show();
    }
}

/**
 * 格式检测
 * @param value {String} 要检测的值
 * @param type {String} 检测的类型
 * @param isNeedInfo {Boolean} 是否需要返回提示信息
 */
function formatCheck(value, type, isNeedInfo) {
    var result = false,
        info = "错误";

    isNeedInfo = isNeedInfo || false;
    switch (type) {
        //  电话号码
        case "telephone":
            result = value.match(/^(([0\+]\d{2,3}-)?(0\d{2,3})-)(\d{7,8})(-(\d{3,}))?$/) !== null;
            break;
        //  手机号码
        case "mobile":
            result = value.match(/^\d{11}$/) !== null;
            break;
        //  邮编
        case "zipCode":
            result = value.match(/^\d{6}$/) !== null;
            break;
        //    邮箱
        case "email":
            result = value.match(/^([a-zA-Z0-9_\.\-])+@(([a-zA-Z0-9\-])+\.)+([a-zA-Z0-9]{2,4})+$/) !== null;
            break;
        case "image":
            result = value.toLowerCase().match(/\.(?:bmp|jpg|jpeg|png|gif)$/) !== null;
            break;
        case "password":
            result = value.toLowerCase().match(/\w{6,16}/) !== null;
            info = "密码长度不符，必须在6-16个字符之间！";
            break;
        //  提现密码
        case "presentPwd":
            result = value.match(/^(?!([a-zA-Z]+|\d+)$)[a-zA-Z\d]{6,16}$/) !== null;
            info = "提现密码必须是6~16个字符，英文+数字组合";
            break;
        //  支付密码
        case "payPwd":
            result = value.match(/\w{6,16}/) !== null;
            info = "密码长度不符，必须在6-16个字符之间！";
            break;
        //  账号名
        case "username":
            result = value.match(/^[a-zA-Z0-9]+$/) !== null;
            info = "用户名错误";
            break;
        //  昵称
        case "nickname":
        //  供应商名称
        case "supplierName":
            result = value.match(/^(\w|[\u4E00-\u9FFF])+$/) !== null;
            info = "名称错误";
            break;
        //    短信验证码
        case "mobileCode":
            result = value.match(/[\w\W]+/) != null;
            break;
        default:
            result = true;
            break;
    }
    return isNeedInfo ? {
        status: result,
        msg: info
    } : result;
}

/**
 * 购物车勾选改变事件
 */
function cartCheckChange() {
    var cartTotalNum = 0 //   勾选的总金额
        , cartCheckNum = 0  //  勾选的总数量
    ;
    $('.CartProChecked.cur').each(function () {
        if (!$(this).hasClass("disabled")) {
            var sid = $(this).attr("data-sid");

            cartTotalNum += Number($(this).parents(".CartLi").find(".CartZPrice i").text() || $(this).parents(".CartLi").find(".CartZPrice i").text()) * $(this).parents(".CartLi").find(".changeAmount").val() || 0;
            cartCheckNum += Number($("#amount" + sid).val());
        }
    });
    $("#CartTotalNum").text(cartTotalNum.toFixed(2));
    $("#CartCheckNum").text(cartCheckNum);
}

// 拼团详情页弹出层
function spellfrome() {
    $("#bg").css({
        display: "block", height: $(document).height()
    });
    $('.spellMask').css({
        display: "block"
    });
    //点击关闭按钮的时候，遮罩层关闭
    $("#bg").on('click', function () {
        $("#bg,.spellMask").css("display", "none");
    });
}

// 拼团详情页文字滚动
function newsContainer() {
    $('#news-container').vTicker({
        speed: 500,
        pause: 3000,
        animation: 'fade',
        mousePause: false,
        showItems: 3
    });
}

// 拼团详情页点击打开详情
function grClick() {
    $(".gr_me_tle").on("click", function () {
        $(".groupMember").toggleClass("open");
    });
}

//  主控制器
if (typeof MainController !== "function") {
    function MainController() {

    }
}

//  页面相关
MainController.page = {
    //  商品详情页
    goods: function () {
        var goodsPackageBtnId = "choosePackage" //  套餐按钮id
            , goodsPackageBoxId = "GoodsRecomBox" //  套餐容器id
            //  开启关闭遮罩层
            , windowBox = {
                show: function () {
                    $('.PsFoot').hide();
                    $('.ChBoxShadow').show();
                    $('main').addClass('mainHidden');
                },
                hide: function () {
                    $('.PsFoot').show();
                    $('.ChBoxShadow').hide();
                    $('main').removeClass('mainHidden');
                }
            };

        //  选择套餐
        if (document.getElementById(goodsPackageBtnId)) {
            $("#" + goodsPackageBtnId).click(function () {
                windowBox.show();
                $("#" + goodsPackageBoxId).show();
            });
            $('#' + goodsPackageBoxId + ' .GoodsRecomH .name').each(function (index) {
                $(this).on('click', function () {
                    var activeName = "cur";

                    $(this).addClass(activeName).siblings().removeClass(activeName);
                    $('.GoodsRecom .GoodsRecoms').eq(index).show().siblings().hide();
                });
            }).eq(0).trigger("click");
        }
        if ($(".PsDetTabH").length && $("#GoodsDetTab").length) {
            //  商品详情，tab导航栏 随窗口移动
            scrollListener(function (scrollTop) {
                var g = $("#GoodsDetTab");
                var header = $('header');
                scrollTop + header.outerHeight() > g.offset().top ? g.addClass("fixed").find(".PsDetTabH").css("top", !header.hasClass('hide') ? header.height() : 0 + "px") : g.removeClass("fixed");
            });
            var p = $(".PsDetTab"),
                f = $(window).height() - (MainController.application.check() ? 0 : $(".MainFoot").outerHeight() + 42 + $(".PsFoot").height() + $("header").height() + $(".PsDetTabH").height());
            p.each(function () {
                if ($(this).outerHeight() < f) $(this).css("min-height", f - $(".PsDetTabs").outerHeight() + $(".PsDetTabs").innerHeight() + "px");
            });
        }
        //  触发选项栏点击
        $(".btnPsDetTabH").on("click", function () {
            $('.PsDetTabH > div').eq($(this).attr("data-id")).trigger("click");
        });
        //  选项栏点击事件
        $('.PsDetTabH > div').each(function (index) {
            $(this).on('click', function () {
                $(this).addClass('cur').siblings().removeClass('cur');
                $('.PsDetTab').eq(index).show().siblings().hide();
                $('body,html').animate({
                    scrollTop: $('#GoodsDetTab').offset().top - $("header").outerHeight()
                }, 300);
            });
        });
        // 立即购买
        if ($('.FBuyBt').length) {
            $('.FBuyBt').hammer().on('tap', function (e) {
                e.gesture.srcEvent.preventDefault();
                windowBox.show();
                $('.ChoiceBox').show().find('.ChBoxBuyBt').addClass('Disblock');
            });
        }
        //  加入购物车
        if (!$("body").hasClass("page-preorderproduct-details")) {
            if ($('.FAddCartBt').length) {
                $('.FAddCartBt').hammer().on('tap', function (e) {
                    e.gesture.srcEvent.preventDefault();
                    windowBox.show();
                    $('.ChoiceBox').show().find('.ChBoxAddCartBt').addClass('Disblock');
                    $("#isSpellPrice").val("0");
                });
            }
        }
        //  选择规格
        $('#liChooseSpecifications').click(function () {
            windowBox.show();
            $('.ChoiceBox').show().find('.ChBoxChoiceBt').addClass('Disblock');
        });
        //  关闭规格和数量选择盒子
        $('.ChBoxShadow, .ChoiceBoxClose').click(function () {
            windowBox.hide();
            $('#' + goodsPackageBoxId).hide();
            $('.ChoiceBox').hide().removeClass("detail-bargain-box").find('.Disblock').removeClass('Disblock');
        });
        //减
        $(".Minus").click(function () {
            if (!$(this).hasClass("Crowdfund") && !$(this).hasClass("prepro")) {
                if ($(this).siblings("input").val() == "" || $(this).siblings("input").val() <= 0) {
                    $(this).siblings("input").val(0)
                }
                var KingMum = parseInt($(this).siblings("input").val());
                if (KingMum == "" || KingMum <= 0) {
                    KingMum = 0;
                } else {
                    KingMum--;
                }
                $(this).siblings("input").val(KingMum);
                $(this).siblings("input").val($(this).siblings("input").val().replace(/[^\d]/g, ''));
            }
        });
        //加
        $(".Plus").click(function () {
            if (!$(this).hasClass("Crowdfund") && !$(this).hasClass("prepro")) {
                if ($(this).siblings("input").val() == "" || $(this).siblings("input").val() <= 0) {
                    $(this).siblings("input").val(0)
                }
                var KingMum = parseInt($(this).siblings("input").val());
                var Maximum = parseInt($(this).siblings(".Maximum").text());
                if (KingMum >= Maximum) {
                    KingMum = Maximum;
                } else {
                    KingMum++;
                }
                $(this).siblings("input").val(KingMum);
                $(this).siblings("input").val($(this).siblings("input").val().replace(/[^\d]/g, ''));
            }
        });
    }
};

//  接口
MainController.api = {
    goods: "/ashx/cn/goods.ashx"    //  商品列表
};

MainController.script = {
    weChatJsSDK: "https://res.wx.qq.com/open/js/jweixin-1.3.2.js"   //  微信JS SDK地址
};

//  app相关
MainController.application = {
    /**
     * 初始化
     */
    init: function () {
        $('.tab-footer').addClass('hide');
        $('header').not('.searchH').addClass('hide');
        $('header.searchH').find('.Hsearchl').addClass('hide');
        $('main').css({"padding-top": 0});
        // if (!$("header").hasClass('PsFoot')) $('main').css({"padding-bottom": 0});
        // cc.addClass("hide");
    },
    /**
     * 检查是否在新版app里
     * @returns {boolean} 是否在新版app里
     */
    check: function () {
        return window.navigator.userAgent.indexOf("Html5Plus") > -1;
    },
    plusReady: function (callback) {
        if (window.plus) {
            callback();
        } else {
            window.addEventListener("plusready", function () {
                callback();
            });
        }
    },
    /**
     * 通知app更新购物车数量
     * @param {Number} sum 购物车数量
     */
    appCartChangeHandler: function (sum) {
        if (MainController.application.check()) {
            MainController.application.plusReady(function () {
                plus.webview.getLaunchWebview().evalJS("if(window.EventEmitter) window.EventEmitter.dispatch('cartSumUpdate'," + sum + ")");
            });
        }
    }
};

//  添加脚本
MainController.loadScript = function (options) {
    var scriptJWeiXin = document.createElement("script")
    ;

    scriptJWeiXin.src = options.url;
    scriptJWeiXin.id = options.id || "";
    scriptJWeiXin.onload = function (ev) {
        options.success();
    };
    scriptJWeiXin.onerror = function (ev) {
        options.error();
    };
    document.body.appendChild(scriptJWeiXin);
};

//  微信小程序相关
MainController.weChatMiniProgram = {
    /**
     * 初始化处理
     * @param completeCB {function} 完成回调
     */
    init: function (completeCB) {

        completeCB = completeCB || function () {

        };

        //  隐藏分佣中心首页的无关二维码
        $(".for-we-chat").addClass("hide");
        //  登陆、会员信息页面，隐藏第三方登陆
        if ($("body").hasClass("page-login")) {
            $(".ContainerLoginOther").addClass("hide");
        }
        if ($("body").hasClass("member-msg")) {
            $("#oauth_app_list").addClass("hide");
        }
        //  支付方式页面，隐藏除了微信以外的第三方支付
        if ($(".PayType .item_pay_bank, .RechargePayTypeList .item_pay_bank").length) {
            $(".PayType .item_pay_bank, .RechargePayTypeList .item_pay_bank").each(function () {
                if (this.querySelector("[name=pay_bank]").value !== "weixinPay") {
                    this.parentNode.removeChild(this);
                }
            });
        }
        //  添加api
        MainController.loadScript({
            url: MainController.script.weChatJsSDK,
            success: function () {
                completeCB(true);
            },
            error: function () {
                completeCB(false);
            }
        });
    },
    /**
     * 检查是否在微信小程序里
     * @return def {Promise} 是否在微信小程序里
     */
    checks: function () {
        var def = $.Deferred()
            , finishCB = function (status) {
            status === true ? def.resolve() : def.reject();
        }
            , ready = function () {
            finishCB(window.__wxjs_environment === 'miniprogram');
        };

        if (MainController.inWechat()) {
            if (typeof wx === "object" && typeof wx.miniProgram === "object" && typeof wx.miniProgram.getEnv === "function") {
                wx.miniProgram.getEnv(function (res) {
                    finishCB(res.miniprogram);
                });
            } else if (!window.WeixinJSBridge || !WeixinJSBridge.invoke) {
                document.addEventListener('WeixinJSBridgeReady', ready, false);
            } else {
                ready();
            }
        } else {
            ready();
        }

        return def.promise();
    },
    /**
     * 设置客服按钮
     * @param btnId {string} 按钮唯一标识
     * @param options {object} 选项
     */
    setCustomService: function (btnId, options) {
        MainController.weChatMiniProgram.checks()
            .done(function () {
                document.getElementById(btnId).addEventListener("click", function (ev) {
                    wx.miniProgram.navigateTo({
                        url: "../customService/customService?from=mweb&data=" + JSON.stringify(options.phoneData) + "&siteMsgOn=" + (options.siteMsgOn === undefined ? "true" : options.siteMsgOn) + "&leaveMsgOn=" + (options.leaveMsgOn === undefined ? "false" : options.leaveMsgOn),
                        success: function () {

                        },
                        fail: options.error
                    });
                });
            })
            .fail(function () {
                var btnCustomService = document.getElementById(btnId);

                if (options.phoneData.IsShow) {
                    btnCustomService.href = "tel:" + options.phoneData.EnTitle;
                } else {
                    btnCustomService.parentNode.removeChild(btnCustomService);
                }
            });
    }
};

//  是否支付宝浏览器
MainController.inALiPay = function () {
    return window.navigator.userAgent.indexOf('AlipayClient') > -1;
};

//  是否微信浏览器
MainController.inWechat = function () {
    return window.navigator.userAgent.indexOf('MicroMessenger') > -1;
};

MainController.weChatConfigReady = function (callback) {
    if (MainController.weChatConfigReady.timer === undefined) {
        MainController.weChatConfigReady.FnArray = [];
        MainController.weChatConfigReady.timer = setInterval(function () {
            if (MainController.wxConfig) {
                clearInterval(MainController.weChatConfigReady.timer);
                MainController.weChatConfigReady.FnArray.forEach(function (callback) {
                    callback();
                });
                MainController.weChatConfigReady.timer = undefined;
            }
        }, 16);
    }
    MainController.weChatConfigReady.FnArray.push(callback);
};

/**
 * 微信自带接口附加完成事件
 * @param callback {function} 完成回调
 */
MainController.weChatBridgeReady = function (callback) {
    if (typeof window.WeixinJSBridge === "object") {
        callback();
    } else {
        document.addEventListener("WeixinJSBridgeReady", callback);
    }
};

//  微信分享初始化
MainController.wechatShareInit = function () {
    //  设置分享
    var scriptId = "jweixin"
        , oldScript = document.getElementById(scriptId)
        , script = document.createElement("script")
        , imageUrl = null   //  要返回的图片路径
        , image = null  //  图片标签
        , dataName = "data-status"
        , dataValue = "success"
    ;

    /**
     * 获取容器中的图片
     * @param containerStr {string} 标签
     * @return {string} 图片路径
     */
    function getImg(containerStr) {
        var container //  容器标签对象
        ;
        if (containerStr) {
            container = document.querySelector(containerStr);
            if (container) {
                if (container.tagName !== "IMG") {
                    container = container.querySelector("img");
                }
                if (container) {
                    if (container.getAttribute("data-src")) {
                        imageUrl = container.getAttribute("data-src");
                    } else {
                        imageUrl = container.src;
                    }
                    imageUrl = (imageUrl.indexOf("http") > -1 ? "" : location.origin) + imageUrl;
                }
            }
        }
        return imageUrl;
    }

    //  等待图片和脚本加载完成的回调
    function promiseCallback() {
        var pubConfigObj = {
            title: document.title, // 分享标题
            desc: MainController.getDesc() || document.title, // 分享描述
            link: encodeURI(decodeURI(location.href)), // 分享链接，该链接域名或路径必须与当前页面对应的公众号JS安全域名一致
            imgUrl: imageUrl, // 分享图标
            success: function () {
                // 用户确认分享后执行的回调函数
            },
            cancel: function () {
                // 用户取消分享后执行的回调函数
            }
        };

        if (imageUrl !== null && script.getAttribute(dataName) === dataValue) {
            setWxConfig(location.href, {}, function (returnConfig) {
                MainController.wxConfig = returnConfig;
                wx.ready(function () {
                    wx.onMenuShareTimeline({
                        title: pubConfigObj.title
                        , link: pubConfigObj.link
                        , imgUrl: pubConfigObj.imgUrl
                        , success: pubConfigObj.success
                        , cancel: pubConfigObj.cancel
                    });
                    wx.onMenuShareAppMessage({
                        title: pubConfigObj.title
                        , desc: pubConfigObj.desc
                        , link: pubConfigObj.link
                        , imgUrl: pubConfigObj.imgUrl
                        , success: pubConfigObj.success
                        , cancel: pubConfigObj.cancel
                    });
                });
                wx.error(function (error) {
                    console.log("微信JSSDK初始化错误：" + JSON.stringify(error));
                });
            });
        }
    }

    if (typeof wechatShare === "boolean" && wechatShare === false) return false;    //  不需要设置微信分享的页面，目前用于商品列表页初次加载时。
    imageUrl = getImg("#wechatShareImg") || /*getImg("#slider") || */getImg("#Psslider");    //  在商品详情页使用商品第一张缩略图
    image = new Image();
    image.onload = function () {
        imageUrl = image.src;
        promiseCallback();
    };
    image.src = imageUrl ? imageUrl : (location.origin + "/images/public/weixin/logo.png");    //  其他页面直接调用logo的固定路径进行加载
    //  在网址改变的情况下将已加载的旧脚本移除
    if (oldScript) {
        oldScript.parentNode.removeChild(oldScript);
    }
    MainController.loadScript({
        id: scriptId,
        url: MainController.script.weChatJsSDK,
        success: function () {
            script.setAttribute(dataName, dataValue);
            promiseCallback();
        },
        error: function () {

        }
    });
};

//  历史操作
MainController.history = {
    /**
     * 网址替换封装
     * @param {Object} data
     * @param {string} title
     * @param {string} url
     */
    replaceState: function (data, title, url) {
        history.replaceState(data, title, url);
        MainController.url.change();
    }
};

//  当前网址处理
MainController.url = {
//  网址改变处理事件
    change: function () {
        if (MainController.inWechat()) {
            if (typeof wechatShare === "boolean") {
                wechatShare = true;
            }
            MainController.wechatShareInit();
        }
    }
};

//  获取页面信息
MainController.getMeta = function (name) {
    var meta = document.getElementsByTagName('meta')
        , content = ''
    ;

    name = name.toLowerCase();
    for (i in meta) {
        if (typeof meta[i].name != "undefined" && meta[i].name.toLowerCase() === name) {
            content = meta[i].content;
        }
    }

    return content;
};

//  获取页面描述
MainController.getDesc = function () {
    return MainController.getMeta("description");
};