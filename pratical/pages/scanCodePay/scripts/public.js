//初始化验证表单
$.fn.initValidform = function (options) {
    if (options == undefined) {
        options = {};
    }
    if (options.beforeSubmit == undefined) {
        options.beforeSubmit = function () {
            return true;
        };
    }
    if (options.ignoreHidden == undefined) {
        options.ignoreHidden = false;
    }
    var checkValidform = function (formObj) {
        $(formObj).Validform({
            beforeSubmit: options.beforeSubmit,
            ignoreHidden: options.ignoreHidden,
            ajaxPost: options.ajaxPost,
            tipMethod: options.tipMethod,//提示方式,氛围artDialog(art)和lhgdialog(lhg)默认lhg
            callback: options.callback,
            tiptype: function (msg, o, cssctl) {
                /*msg：提示信息;
                o:{obj:*,type:*,curform:*}
                obj指向的是当前验证的表单元素（或表单对象）；
                type指示提示的状态，值为1、2、3、4， 1：正在检测/提交数据，2：通过验证，3：验证失败，4：提示ignore状态；
                curform为当前form对象;
                cssctl:内置的提示信息样式控制函数，该函数需传入两个参数：显示提示信息的对象 和 当前提示的状态（既形参o中的type）；*/
                //全部验证通过提交表单时o.obj为该表单对象;
                if (!o.obj.is("form")) {
                    ////定位到相应的Tab页面
                    //if (o.obj.is(o.curform.find(".Validform_error:first"))) {
                    //    var tabobj = o.obj.parents(".tab-content"); //显示当前的选项
                    //    var tabindex = $(".tab-content").index(tabobj); //显示当前选项索引
                    //    if (!$(".content-tab ul li").eq(tabindex).children("a").hasClass("selected")) {
                    //        $(".content-tab ul li a").removeClass("selected");
                    //        $(".content-tab ul li").eq(tabindex).children("a").addClass("selected");
                    //        $(".tab-content").hide();
                    //        tabobj.show();
                    //    }
                    //}
                    ////页面上不存在提示信息的标签时，自动创建;
                    //if (o.obj.parent().find(".Validform_checktip").length == 0) {
                    //    o.obj.parent().append("<span class='Validform_checktip' />");
                    //    o.obj.parent().next().find(".Validform_checktip").remove();
                    //}
                    var objtip = o.obj.siblings(".Validform_checktip");
                    cssctl(objtip, o.type);
                    if (o.type != 2) {
                        if (options.tipMethod == "art") {
                            art.dialog.tips(msg, 1.5);
                        } else {
                            var parent1 = $(this).find(".ui_border").parent().css("visibility");
                            var parent2 = $(parent.document).find(".ui_border").parent().css("visibility");
                            //console.log(parent1+"  -  "+parent2);
                            if ((parent1 != undefined && parent1 != "visible") || (parent2 != undefined && parent2 != "visible")) {
                                $.dialog.tips(msg, 1.5, '32X32/fail.png');
                            }
                        }
                        objtip.text(msg);
                    } else {
                        objtip.text('');
                    }
                }
            },
            datatype: {
                "decimal": function (gets, obj, curform, regxp) {
                    return /^-?\d+\.\d+$/.test(gets) || /^-?\d+$/.test(gets);
                }
            },
            showAllError: true
        });
    };
    return $(this).each(function () {
        checkValidform($(this));
    });
}
//========================基于lhgdialog插件========================
//可以自动关闭的提示，基于lhgdialog插件
function jsprint(msgtitle, url, msgcss, obj, callback) {
    var iconurl = "";
    switch (msgcss) {
        case "Success":
            iconurl = "32X32/succ.png";
            break;
        case "Error":
            iconurl = "32X32/fail.png";
            break;
        default:
            iconurl = "32X32/hits.png";
            break;
    }
    $.dialog.tips(msgtitle, 2, iconurl);
    if (url == "back") {
        //frames["mainframe"].history.back(-1);
    } else if (url != "") {
        if (frames["mainframe"] == undefined) {
            window.location.href = url;
        }
        //else {
        //    frames["mainframe"].location.href = url;
        //}
    }
    if (obj != undefined && obj != "") {
        $(obj).select();
        $(obj).focus();
    }
    //执行回调函数
    if (arguments.length == 5) {
        callback();
    }
}

//弹出一个Dialog窗口
function jsdialog(msgtitle, msgcontent, url, msgcss, callback) {
    var iconurl = "";
    var argnum = arguments.length;
    switch (msgcss) {
        case "Success":
            iconurl = "success.gif";
            break;
        case "Error":
            iconurl = "error.gif";
            break;
        default:
            iconurl = "alert.gif";
            break;
    }
    var dialog = $.dialog({
        title: msgtitle,
        content: msgcontent,
        fixed: true,
        min: false,
        max: false,
        lock: true,
        icon: iconurl,
        ok: true,
        close: function () {
            if (url == "back") {
                history.back(-1);
            } else if (url != "") {
                location.href = url;
            }
            //执行回调函数
            if (argnum == 5) {
                callback();
            }
        }
    });
}

//打开一个最大化的Dialog
function ShowMaxDialog(tit, url) {
    $.dialog({
        title: tit,
        content: 'url:' + url,
        min: false,
        max: false,
        lock: false
    }).max();
}

//执行回传函数
function ExePostBack(objId, objmsg) {
    if ($(".checkall input:checked").size() < 1) {
        $.dialog.alert('对不起，请选中您要操作的记录！');
        return false;
    }
    var msg = "删除记录后不可恢复，您确定吗？";
    if (arguments.length == 2) {
        msg = objmsg;
    }
    $.dialog.confirm(msg, function () {
        __doPostBack(objId, '');
    });
    return false;
}


if (typeof DataBaseController !== "function") {
    //  数据库控制
    function DataBaseController() {
        this.dbName = "micronet";
        this.version = 1.0;
    }
}

//  打开数据库
DataBaseController.prototype.open = function (options) {
    var dbController = this
        , indexDB = window.indexedDB
        , request
    ;

    this.dbName = options.dbName || this.dbName;
    this.version = options.version || this.version;
    request = indexDB.open(this.dbName, this.version);
    request.onupgradeneeded = function (e) {
        dbController.upgradeneeded(e);
        if (typeof options.upgradeneeded === "function") {
            options.upgradeneeded(e);
        }
    };
    request.onsuccess = function (e) {
        dbController.db = e.target.result;
        if (typeof options.success === "function") {
            options.success(e);
        }
    };
    request.onerror = function (e) {
        if (typeof options.error === "function") {
            options.error(e);
        }
    };

    return request;
};

//  根据键值获取数据
DataBaseController.prototype.getDataByKey = function (storeName, keyValue, callback) {
    var request = this.open();

    request.onsuccess = function () {
        var db = ev.target.result;

        var transaction = db.transaction(storeName);
        var os = transaction.objectStore(storeName);
        var request = os.get(idValue);
        request.onsuccess = function (e) {
            callback(e.target.result);
        };
        request.onerror = function (e) {
            callback(null);
        }
    }
};

DataBaseController.prototype.saveData = function (storeName, data) {
    var request = this.open();

    request.onsuccess = function (ev) {
        var db = ev.target.result
            , transactionUserInfo = db.transaction(storeName, "readwrite")
            , objectStore = transactionUserInfo.objectStore(storeName)
        ;

        objectStore.add(data);
    };
};

DataBaseController.prototype.upgradeneeded = function (e) {
    var db = e.target.result;

    if (e.newVersion > e.oldVersion) {
        var userOS = db.createObjectStore("user", {keyPath: "id"});
        userOS.createIndex("userNameIndex", "username", {
            unique: true
        });
        // db.createIndex("user");
        var goodsOS = db.createObjectStore("goods", {keyPath: "id"});
        goodsOS.createIndex("saleUp", "salePrice", {
            unique: false
        });
    }
};

if (typeof PublicController !== "function") {
    //  公共控制
    function PublicController() {

    }
}

//  公共设置
PublicController.globalOptions = {
    pageSize: 20    //  每页数量
};

//  接口
PublicController.api = {
    mobile: "/ashx/cn/mobile.ashx" //  检测手机是否已注册
    , commentImgUpload: '/api/home/CommentImgUpload'    //  评论图片上传
    , region: "/ashx/cn/region.ashx"    //  获取地区
    , receivingAddress: "/ashx/cn/shopping_address.ashx"    //  收货地址
    , specifications: "/ashx/cn/specifications.ashx"
};

PublicController.adapter = {
    addReceivingAddress: function (data) {
        return data;
    },
    supplierList: function (data) {
        var _data = data.map(function (item) {
            if (item.region === "") {
                item.region = "未知";
            }
        });

        return data;
    },
    region: function (data) {
        data = data || [];

        return data.map(function (item) {
            if (item.Id) {
                item.id = item.Id;
                delete item.Id;
            }
            if (item.RegionName) {
                item.regionName = item.RegionName;
                delete item.RegionName;
            }

            return item;
        });
    }
};

PublicController.provider = {
    //  获取商品实体信息
    getSpecByGoodsId: function (options) {
        options.url = options.url || PublicController.api.specifications;
        options.type = options.type || "POST";
        options.data = "goods=" + options.data.goodsId;
        PublicController.methods.request(options);
    },
    //  添加收货地址
    addReceivingAddress: function (options) {
        var _success = options.success;

        options.url = options.url || PublicController.api.receivingAddress;
        options.type = options.type || "POST";
        options.data = encodeURI("consignee=" + options.data.consignee + "&region=" + options.data.region + "&address=" + options.data.address + "&zipcode=" + options.data.zipCode + "&telephone=" + (options.data.telephone || "") + "&mobile=" + options.data.mobile + "&isDefault=" + options.data.isDefault + "&id=" + options.data.id);
        options.success = function (data) {
            var delay = 1.5
                , handler = {
                "1000": function (data) {
                    art.dialog.tips("添加成功", delay);
                },
                "1001": function (data) {
                    art.dialog.tips("添加失败", delay);
                },
                "1002": function (data) {
                    art.dialog.tips("更新成功", delay);
                },
                "1003": function (data) {
                    art.dialog.tips("更新失败", delay);
                }
            };

            data = PublicController.adapter.addReceivingAddress(data);
            if (art.dialog.list['Tips']) art.dialog.list['Tips'].close();
            handler[data.status](data);
            _success(data);
        };
        PublicController.methods.request(options);
    },
    //  判断手机是否已经注册账号
    checkRegister: function (options) {
        options.url = options.url || PublicController.api.mobile;
        options.type = options.type || "POST";
        options.data = "type=checkmobile&mobile=" + options.mobile;
        PublicController.methods.request(options);
    },
    //  发送短信验证码
    sendCode: function (options) {
        options.url = options.url || PublicController.api.mobile;
        options.type = options.type || "POST";
        options.data = "type=send&mobile=" + options.data.mobile;
        PublicController.methods.request(options);
    },
    //  获取地区
    getRegion: function (options) {
        options.url = options.url || PublicController.api.region;
        options.type = options.type || "POST";
        options.data = "pid=" + options.data.regionId;
        PublicController.methods.request(options);
    }
};

//  路由
PublicController.route = {
    /**
     *  跳转
     * @param routeName {string} 路由名称
     * @param queries {object} 请求参数
     */
    go: function (routeName, queries) {
        location.href = PublicController.route.getPathByRouteName(routeName, queries);
    },
    /**
     *  用路由名称获取路径
     * @param routeName {string} 路由名称
     * @param queries {object} 请求参数
     */
    getPathByRouteName: function (routeName, queries) {
        var link = ""
            , queryStr = ""
            , preFix = ""
        ;

        queries = queries || {};
        for (var keyName in queries) {
            if (queries.hasOwnProperty(keyName)) {
                queryStr += "&" + keyName + "=" + queries[keyName];
            }
        }
        link = routeName + queryStr.replace(/^&/, "?");

        if (routeName.indexOf("/backstage") === -1) {
            preFix = PublicController.methods.isMobileOrPad()
                ? PublicController.route.mobilePreFix
                : PublicController.route.pcPreFix;
        }

        return preFix + link;
    },
    getPathParams: function (path) {
        return path.match(/{[\w\W]+}/g);
    },
    mobilePreFix: "/mobi/cn",
    pcPreFix: "/cn",
    link: {
        //  商品
        goods: {
            list: "/goods_list.html"  //  列表
        },
        //  供应商
        supplier: {
            agreement: "/supplier_register_agreement.html"  //  用户协议
            , register: "/supplier/register.html"  //  注册页
            , login: "/supplier/register.html"  //  登录页（与注册共用）
            , list: "/supplier_list.html"  //  列表
        },
        user: {
            receivingAddressList: "/member/shipping/address_0.html" //  添加收货地址
            , couponList: "/member/coupon/list/1.html"
            , memberCenter: "/member/"
            , prizeList: "/mobi/turnTable/prizeList.html"
        },
        backstage: {
            login: "/backstage/login.aspx" //  后台登陆页
        }
    }
};

//  公用方法
PublicController.methods = {
    //  加载器
    loader: {
        //  图片预加载
        img: function (arr, callback) {
            var handler = function (options) {
                    var image = new Image();

                    image.onload = options.success;
                    image.onerror = options.error;
                    image.src = options.src;
                }
                , readyNum = 0
                , completeCB = function (index, result) {
                    arr[index] = result;
                    if (++readyNum === arr.length) {
                        callback(arr);
                    }
                }
            ;

            arr.forEach(function (src, index) {
                handler({
                    src: src,
                    success: function () {
                        completeCB(index, true);
                    },
                    error: function () {
                        completeCB(index, false);
                    }
                });
            });
        }
    },
    //  是否为手机/平板
    isMobileOrPad: function () {
        // return /Android|webOS|iPhone|iPod|BlackBerry/i.test(navigator.userAgent);
        return /android.+mobile|avantgo|bada\/|blackberry|blazer|micronetapp|micromessenger|compal|elaine|fennec|hiptop|iemobile|ip(hone|[oa]{1}d)|iris|kindle|lge |maemo|midp|mmp|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|symbian|treo|up\.(browser|link)|vodafone|wap|windows (ce|phone)|xda|xiino|ucweb|mqqbrowser/i.test(navigator.userAgent);
    },
    //  请求
    request: function (options) {
        $.ajax({
            type: options.type || "POST",
            dataType: options.dataType || "JSON",
            url: options.url,
            data: options.data,
            success: options.success,
            error: options.error
        });
    },
    /**
     * 异步上传文件（依赖ajaxfileupload.js）
     * @param options {object} 选项
     */
    uploadImg: function (options) {
        options.statusChange = options.statusChange || function () {

        };
        options.success = options.success || function () {

        };
        options.error = options.error || function () {

        };
        if (!formatCheck(options.value, options.fileType || "image")) {
            return options.statusChange(0);
        }
        options.statusChange(1);
        $.ajaxFileUpload({
            url: options.url, //用于文件上传的服务器端请求地址
            secureuri: false, //一般设置为false
            fileElementId: options.id, //  文件上传空间的id属性  <input type="file" id="file" name="file" />
            dataType: 'json', //返回值类型 一般设置为json
            data: {
                fileId: options.name || options.id
            },
            success: function (apiData, status) {
                var apiDataObj = eval("(" + apiData + ")");

                options.statusChange(2, apiDataObj, status);
                options.success(apiDataObj, status);
            },
            error: function (data, status, e) {
                options.statusChange(3, data, status, e);
                options.error(data, status, e);
            },
            complete: options.complete || function (xml, status) {
            }
        });
    },
    //  对象继承
    objectAssign: function (oldObj, newsObj) {
        var finallyObj = newsObj instanceof Array ? [] : {};    //  最终对象

        if (oldObj) {
            finallyObj = PublicController.methods.objectAssign(null, oldObj);
        }
        if (newsObj) {
            for (var p in newsObj) {
                if (newsObj.hasOwnProperty(p)) {
                    if (typeof newsObj[p] === "object" || newsObj[p] instanceof Array) {
                        finallyObj[p] = PublicController.methods.objectAssign(finallyObj[p], newsObj[p]);
                    } else {
                        finallyObj[p] = newsObj[p];
                    }
                }
            }
        }

        return finallyObj;
    },
    /**
     * 更新购物车显示数量
     * @param value
     */
    updateCartAmount: function (value) {
        var container = $("#shoppingCartAmount")
            , subNumClass = "sub-num"
            , targetNode = container.find("." + subNumClass)
        ;

        if (value > 99) {
            value = 99;
        }
        value === 0 ? targetNode.remove() : (targetNode.length ? targetNode.html(value) : container.append("<span class='" + subNumClass + "'>" + value + "</span>"));
        if (typeof MainController.application.appCartChangeHandler === 'function') MainController.application.appCartChangeHandler(value);
    },
    //  通过微信 JS-SDK 获取地理坐标
    getLocationByWeChat: function (options) {
        wx.ready(function () {
            wx.getLocation({
                type: 'wgs84', // 默认为wgs84的gps坐标，如果要返回直接给openLocation用的火星坐标，可传入'gcj02'
                success: function (res) {
                    options.success(res);
                },
                cancel: function (res) {
                    // alert('用户拒绝授权获取地理位置');
                    options.error(res);
                }
            });
        });
    }
};

function Modal() {
    this.modalTitle = "modal-title";
    this.slot = "slot";
}

//  初始化
Modal.prototype.init = function (options) {
    var container = document.getElementById(options.id)
        , self = this
        , containerClass = "container-modal"
        , controlClass = "btns"
    ;

    options.buttons = options.buttons || [];
    if (!container) {
        container = document.createElement("div");
    }
    this.hideClassName = options.hideClassName || "hide";
    container.id = this.id = options.id;
    container.innerHTML = '<div class="' + containerClass + ' ' + (options.containerClass || "") + '"><div class="cover"></div><div class="show-pro result"><div class="window"><div class="' + this.modalTitle + '">' + (options.title || "") + '</div><div class="content"><div class="' + this.slot + '">' + (options.content || "") + '</div></div><div class="' + controlClass + '"></div></div></div></div>';
    options.buttons.forEach(function (item, index) {
        var button = document.createElement("a");

        button.href = "#";
        button.className = "btn " + item.className;
        button.onclick = function () {
            options.noCloseAuto ? "" : self.hide();
            options.callback.call(self, index, document.getElementById(self.id));

            return false;
        };
        button.innerText = item.title;
        container.querySelector("." + controlClass).appendChild(button);
    }, "");
    document.body.appendChild(container);
    this.hide();
};

//  显示
Modal.prototype.show = function (options) {
    var container = document.getElementById(this.id);

    options = options || {};
    if (options.content) {
        container.querySelector("." + this.slot).innerHTML = options.content;
    }
    container.classList.remove(this.hideClassName);
};

//  隐藏
Modal.prototype.hide = function () {
    document.getElementById(this.id).classList.add(this.hideClassName);
};

if (typeof GoodsCartController !== "function") {
    //  商品购物车控制器
    function GoodsCartController() {

    }
}

//  初始化
GoodsCartController.prototype.init = function (options) {
    var vue
        , computed = {
            checkedSum: function () {
                var sum = 0
                ;

                this.cart.list.forEach(function (item) {
                    if (item.checked) {
                        sum += parseFloat(item.goodsPackage === 0 ? item.total : item.packageTotal);
                    }
                });

                return sum;
            },
            //  选中总数量
            checkedCount: function () {
                var count = 0
                ;

                this.cart.list.forEach(function (item) {
                    if (item.checked) {
                        count += item.amount;
                    }
                });

                return count;
            },
            //  是否已全选
            isSelectedAll: function () {
                var isAll = true;

                this.cart.list.forEach(function (item) {
                    if (!item.checked) {
                        isAll = false;

                        return false;
                    }
                });

                return isAll;
            }
        }
        , data = {
            cart: {
                statistics: {},
                sumGiveIntegral: 0,
                list: []
            }
        }
        , methods = {
            //  获取商品详情链接
            getDetailLinkByGoodsId: function (id) {
                return "/cn/goods/" + id + ".html";
            },
            //  状态是否正常
            isStatusNormalByIndex: function (index, item) {
                item = item || this.cart.list[index];

                return item.amount <= item.inventory && item.amount > 0 && item.isShelves === 1;
            },
            //  全选/反选
            selectAll: function () {
                var vue = this
                    , isAll = this.isSelectedAll
                ;

                this.cart.list.forEach(function (item, index) {
                    if (vue.isStatusNormalByIndex(index)) {
                        item.checked = !isAll;
                    }
                });
            },
            //  选择一个
            checkOneByIndex: function (index) {
                if (vue.isStatusNormalByIndex(index)) {
                    this.cart.list[index].checked = !this.cart.list[index].checked;
                }
            },
            //  删除商品
            deleteGoodsById: function (sid) {
                var vue = this;

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
                                var data = "sid=" + sid;

                                vue.changeShoppingCart(data);
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
            },
            //  删除套餐
            deletePackageById: function (packageId) {
                var vue = this;

                art.dialog({
                    id: 'testID',
                    content: "确定要删除吗？",
                    lock: true,
                    fixed: true,
                    opacity: 0.1,
                    button: [
                        {
                            name: "确定",
                            callback: function () {
                                var data = 'gpid=' + packageId;

                                vue.changeShoppingCart(data);
                            },
                            focus: true
                        },
                        {
                            name: "取消",
                            callback: function () {
                            }
                        }
                    ]
                });
            },
            //  减少数量
            minusNumByIndex: function (index) {
                var item = this.cart.list[index]
                    , sid = item.sid
                ;

                if (item.amount > 1) {
                    this.changeShoppingCart("cid=" + sid + "&amount=" + (item.amount - 1));
                }
            },
            //  增加数量
            plusNumByIndex: function (index) {
                //  购物车，商品数量加一
                var item = this.cart.list[index]
                    , sid = item.sid
                    , data = "cid=" + sid + "&amount=" + (item.amount + 1)
                ;

                this.changeShoppingCart(data);
            },
            //  修改数量
            changeNumByIndex: function (index) {
                //  购物车，直接填写商品数量
                var item = this.cart.list[index]
                    , sid = item.sid
                ;

                if (item.amount === 0) {
                    art.dialog.tips("购买数量不能为0！", 1.5);
                    return false;
                } else {
                    var data = "cid=" + sid + "&amount=" + $("#amount" + sid).val();
                    vue.changeShoppingCart(data);
                }
            },
            //  删除所选商品
            deleteChecked: function () {
                var vue = this
                    , checkedGoods = []
                    , checkedPackage = []
                ;

                this.cart.list.forEach(function (item) {
                    if (item.checked) {
                        if (this.goodsPackage) {
                            if (this.packageName) {
                                checkedPackage.push(item.goodsPackage);
                            }
                        } else {
                            checkedGoods.push(item.sid);
                        }
                    }
                });
                if (checkedPackage.length > 0 || checkedGoods.length > 0) {
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
                                    var data = ('gpid=' + checkedPackage.join(",") + "&sid=" + checkedGoods.join(",")).replace(/^&/, "");

                                    vue.changeShoppingCart(data);
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
            },
            //  购物车，增删加减改都调这个
            changeShoppingCart: function (data, callback) {
                var vue = this
                    , delay = 1.5;    //  延迟时间

                callback = callback || function () {

                };
                $.ajax({
                    type: "POST",
                    url: "/ashx/cn/shopping_cart.ashx",
                    data: data || "",
                    //data: 'sid=all',//  清空购物车
                    dataType: "json",
                    success: function (obj) {
                        var loadSelectStatus = function (cartList) {
                            var vue = this
                                , tempData = {}
                                , typeGoods = "goods"
                                , typePackage = "package"
                                , goodsIdPropName = "sid"
                                , packageIdPropName = "goodsPackage"
                            ;

                            tempData[typeGoods] = {};
                            tempData[typePackage] = {};
                            this.cart.list.forEach(function (item) {
                                var prop = ""
                                    , type = ""
                                ;

                                if (item[packageIdPropName] === 0) {
                                    prop = goodsIdPropName;
                                    type = typeGoods;
                                } else {
                                    prop = packageIdPropName;
                                    type = typePackage;
                                }
                                tempData[type][item[prop]] = true;
                            });

                            return cartList.map(function (item, index) {
                                var prop = ""
                                    , type = ""
                                ;

                                if (item[packageIdPropName] === 0) {
                                    prop = goodsIdPropName;
                                    type = typeGoods;
                                } else {
                                    prop = packageIdPropName;
                                    type = typePackage;
                                }
                                item.checked = !!tempData[type][item[prop]] && vue.isStatusNormalByIndex(index, item);

                                return item;
                            });
                        };

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
                                location.href = "/cn/login/web.html?redirect=" + location.href;
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
                            case 0:  //  0是删除多个商品成功
                            case 2000:  //  2000是增减改商品数量成功
                            case 3000:  //  3000是删除商品成功
                            default:
                                //  成功，用百度模板重新显示购物车内的商品信息
                                vue.cart.list = loadSelectStatus.call(vue, obj.data.shoppingcart);
                                vue.cart.statistics.sumAmount = obj.data.sumAmount;
                                callback();
                                break;
                        }
                    }
                });
                return false;
            },
            //  购物车 结算按钮
            cartSubmit: function () {
                var vue = this;

                if (this.checkedCount) {
                    PublicController.methods.request({
                        url: "/ashx/cn/detection_session.ashx",
                        success: function (obj) {
                            switch (obj.status) {
                                case 1000:
                                    var data = [];

                                    vue.cart.list.forEach(function (item) {
                                        if (item.checked) {
                                            data.push(item.geid);
                                        }
                                    });
                                    window.location = "/cn/order/submit/" + data.join("_") + ".html";
                                    break;
                                case 1001:
                                    //登录注册弹窗 显示
                                    $('.member_container,.pop_member').show(300);
                                    break;
                                case 1002:
                                    art.dialog.tips("购物车目前没有加入任何商品！", 1.5);
                                    break;
                                default:
                                    break;
                            }
                        },
                        error: function (err) {
                            alert("获取登录状态失败：" + JSON.stringify(err));
                        }
                    });
                } else {
                    art.dialog.tips("请选择商品！", 1.5);
                }

                return false;
            }
        }
    ;

    options.data = PublicController.methods.objectAssign(data, options.data);
    options.data.cart.list = options.data.cart.list.map(function (item) {
        if (item.amount > 0 && item.amount <= item.inventory) {
            item.checked = true;
        }

        return item;
    });
    options.computed = PublicController.methods.objectAssign(computed, options.computed);
    options.methods = PublicController.methods.objectAssign(methods, options.methods);
    vue = new Vue({
        el: options.el,
        data: options.data,
        computed: options.computed,
        methods: options.methods,
        created: function () {
            var vue = this;

            this.changeShoppingCart("", function () {
                vue.selectAll();
            });
            options.created();
        }
    });

    return vue;
};

//  抽奖控制器
function LotteryController() {

}

//  接口地址
LotteryController.apiPrevFix = "";  //  代理

LotteryController.api = {
    generate: LotteryController.apiPrevFix + "/lottery/generate"
    , getRecordByLotteryId: LotteryController.apiPrevFix + "/lottery/record"  //  获取抽奖记录
};

LotteryController.globalOptions = PublicController.globalOptions;

LotteryController.adapter = {
    record: function (apiData) {
        if (apiData.status === 1000) {
            apiData.data = apiData.data.map(function (item) {
                return {
                    type: item.prizeType,
                    nickname: item.userName,
                    avatar: item.userFace,
                    createDate: item.createTime,
                    prizeText: item.prizeName
                };
            });
        }

        return apiData;
    }
};

LotteryController.provider = {
    //  生成抽奖信息
    generateById: function (options) {
        var dataStr = "";

        options.url = options.url || LotteryController.api.generate;
        options.type = options.type || "POST";
        for (var p in options.data) {
            if (options.data.hasOwnProperty(p)) {
                dataStr += "&" + p + "=" + options.data[p];
            }
        }
        options.data = dataStr.replace(/^&/, "");
        PublicController.methods.request(options);
    },
    //  获取抽奖记录
    getRecordByLotteryId: function (options) {
        var _success = options.success;

        options.url = options.url || LotteryController.api.getRecordByLotteryId;
        options.type = options.type || "GET";
        options.data = "lotteryId=" + options.data.id + "&type=" + (options.data.type || 0) + "&pageNum=" + (options.data.pageNum || 1) + "&pageSize=" + (options.data.pageSize || LotteryController.globalOptions.pageSize);
        options.success = function (apiData) {
            _success(LotteryController.adapter.record(apiData));
        };
        PublicController.methods.request(options);
    }
};

LotteryController.vue = {
    //  大转盘活动页
    wheelActivityPage: function (options) {
        var computed = {
                userChanceForToday: function () {
                    return this.game.dayLimit === 0 ? Infinity : this.game.dayLimit - this.user.dayLotteryCount;
                },
                userChanceForAll: function () {
                    return this.game.totalLimit === 0 ? Infinity : this.game.totalLimit - this.user.totalLotteryCount;
                },
                isGameOn: function () {
                    return this.game.configStatus === 2 && (this.game.loadStatus === 1 || this.game.loadStatus === 2);
                },
                gameStatusText: function () {
                    var str = "";

                    switch (this.game.loadStatus) {
                        case 1:
                            str = "加载中";
                            break;
                        case 2:
                            str = "加载完成";
                            break;
                        case 3:
                            str = "错误";
                            break;
                    }
                    switch (this.game.configStatus) {
                        case -1:
                            str = "已关闭";
                            break;
                        case 0:
                            str = "未开始";
                            break;
                        case 1:
                            str = "已结束";
                            break;
                        case 2:
                            str = "进行中";
                            break;
                    }

                    return str;
                }
            }
            , data = {
                delay: 1.5,
                imagePlaceHolder: "",
                user: {
                    //  日已抽奖次数
                    dayLotteryCount: 0,
                    //  总已抽奖次数
                    totalLotteryCount: 0,
                    //  用户当前积分
                    integral: 0
                },
                game: {
                    id: 0,
                    dctId: "",
                    configStatus: 0, // 游戏设置状态【-1：已关闭，0：未开始，1：已结束，2：进行中】
                    loadStatus: 1, //   游戏加载状态【1：正在初始化，2：已完成初始化，3：错误】
                    //  消费积分/次
                    cost: 0,
                    //  日抽奖次数限制
                    dayLimit: 0,
                    //  总抽奖次数限制
                    totalLimit: 0,
                    prizes: [],
                    rule: "",
                    player: {
                        list: [],
                        status: 0,
                        rewardList: [],
                        pageNum: 0
                    }
                }
            },
            methods = {
                bindMobile: function (options) {
                    var vue = this
                        , _success = options.success || function () {

                        }
                        , mobileNumber = options.data.mobileNumber
                    ;

                    options.success = function (apiData) {
                        var handler = {
                            1000: function (apiData) {
                                this.user.mobileNumber = mobileNumber;
                                art.dialog.tips(apiData.msg, this.delay);
                                this.hideResult();
                            }
                        };

                        if (handler[apiData.status]) {
                            handler[apiData.status].call(vue, apiData);
                        } else {
                            art.dialog.tips(apiData.msg, vue.delay);
                        }
                        _success(apiData);
                    };

                    return UserController.methods.updateLotteryMobileNumber(options);
                },
                getRecord: function (options) {
                    var vue = this;

                    options.data = options.data || {};
                    LotteryController.provider.getRecordByLotteryId({
                        data: {
                            id: vue.game.id,
                            type: options.data.type,
                            pageNum: options.data.pageNum,
                            pageSize: options.data.pageSize
                        },
                        success: function (apiData) {
                            var handler = {
                                1000: function (apiData) {
                                },
                                //  接口异常
                                1001: function (apiData) {
                                    art.dialog.tips(apiData.msg, vue.delay);
                                },
                                //  该功能未开启
                                1002: function (apiData) {
                                    art.dialog.tips(apiData.msg, vue.delay);
                                },
                                //  缺少参数
                                1003: function (apiData) {
                                    art.dialog.tips(apiData.msg, vue.delay);
                                },
                                //  找不到指定活动信息
                                1004: function (apiData) {
                                    art.dialog.tips(apiData.msg, vue.delay);
                                },
                                //  没有更多数据
                                1005: function (apiData) {
                                }
                            };

                            vue.game.player.status = 2;
                            handler[apiData.status](apiData);
                            options.success(apiData);
                        },
                        error: function (err) {
                            alert("获取记录出错：" + JSON.stringify(err));
                            options.error(err);
                        }
                    });
                },
                getRewardRecord: function (options) {
                    var vue = this;

                    this.getRecord({
                        data: {
                            type: 1,
                            pageNum: 1
                        },
                        success: function (apiData) {
                            if (apiData.status === 1000) {
                                vue.game.player.rewardList = apiData.data.map(function (item, index) {
                                    var img = item.avatar
                                    ;

                                    PublicController.methods.loader.img([img], function (arr) {
                                        arr[0] ? vue.game.player.rewardList[index].avatar = img : "";
                                    });
                                    item.avatar = vue.imagePlaceHolder;

                                    return item;
                                });

                                vue.game.player.rewardList = apiData.data;
                            }
                            options.success();
                        },
                        error: function (err) {

                        }
                    });
                },
                getNextRecord: function (options) {
                    var vue = this;

                    if (vue.game.player.status === 1 || vue.game.player.status === 3 || vue.game.player.status === 4) {
                        return false;
                    }
                    vue.game.player.status = 1;
                    this.getRecord({
                        data: {
                            type: 0,
                            pageNum: ++vue.game.player.pageNum,
                            pageSize: options.data.pageSize
                        },
                        success: function (apiData) {
                            if (apiData.status === 1000) {
                                vue.game.player.list = vue.game.player.list.concat(apiData.data.map(function (item, index) {
                                    var length = vue.game.player.list.length
                                        , img = item.avatar
                                    ;

                                    PublicController.methods.loader.img([img], function (arr) {
                                        arr[0] ? vue.game.player.list[index + length].avatar = img : "";
                                    });
                                    item.avatar = vue.imagePlaceHolder;

                                    return item;
                                }));
                            } else if (apiData.status === 1005) {
                                vue.game.player.status = 3;
                            }
                        },
                        error: function (err) {
                            vue.game.player.status = 4;
                        }
                    });
                },
                showResult: function (options) {
                    if (!this.resultInstance) {
                        this.resultInstance = new Modal();
                    }
                    this.resultInstance.init({
                        id: "activity-wheel-result",
                        containerClass: "activity-wheel-result",
                        title: options.title,
                        buttons: options.buttons,
                        callback: options.callback,
                        noCloseAuto: options.noCloseAuto
                    });
                    this.resultInstance.show({
                        content: options.content
                    });
                },
                hideResult: function () {
                    this.resultInstance.hide();
                },
                exchangeById: function (id, callback) {
                    var vue = this;

                    return UserController.methods.lotteryPrizeExchangeById(id, 0, {
                        success: callback,
                        error: function (err) {
                            alert("领取奖品出错：" + JSON.stringify(err));
                        }
                    });
                },
                generate: function (options) {
                    var vue = this;

                    this.user.integral -= this.game.cost;
                    this.user.dayLotteryCount++;
                    this.user.totalLotteryCount++;
                    LotteryController.provider.generateById({
                        data: {
                            lotteryId: this.game.id,
                            dctId: this.game.dctId
                        },
                        success: function (apiData) {
                            if (apiData.status !== 1000) {
                                vue.user.integral += vue.game.cost;
                                vue.user.dayLotteryCount--;
                                vue.user.totalLotteryCount--;
                            }
                            options.success(apiData);
                        },
                        error: options.error
                    });
                }
            }
        ;

        options.data = PublicController.methods.objectAssign(data, options.data);
        options.computed = PublicController.methods.objectAssign(computed, options.computed);
        options.methods = PublicController.methods.objectAssign(methods, options.methods);

        return new Vue({
            el: options.el,
            data: options.data,
            computed: options.computed,
            methods: options.methods,
            created: function () {
                options.created.call(this);
            },
            mounted: function () {
                options.mounted.call(this);
            }
        });
    }
};

//  砍价控制器
function BargainController() {

}

BargainController.api = {
    check: "/api/bargain/CheckEntity",
    createActivity: "/api/bargain/CreateActivity"
};

BargainController.methods = PublicController.methods;

//  检查商品实体活动状态和确认活动的失败返回处理
BargainController.methods.checkStatusFailCallback = function (apiData) {
    switch (apiData.ret) {
        case "1003":
            art.dialog.tips("请先登录", 1.5);
            location.href = (PublicController.methods.isMobileOrPad() ? "/mobi/cn/login.html" : "/cn/login/web.html") + "?redirect=" + location.href;
            break;
        case "1002":
            art.dialog.tips("商品库存不足", 1.5);
            break;
        case "-1001":
            art.dialog.tips("砍价活动未开启或该商品规格不参与活动", 1.5);
            setTimeout('location.reload()', 500);
            break;
        case "-1002":
            art.dialog.tips("该商品已有一个活动正在进行中", 1.5);
            setTimeout(function () {
                window.location.href = (PublicController.methods.isMobileOrPad() ? "/mobi" : "") + "/cn/member/bargain/UserActivityDetail.html?Id=" + apiData.data.Id;
            }, 500);
            break;
        default:
            art.dialog.tips("暂不可用", 1.5);
            setTimeout('location.reload()', 500);
            break;
    }
};

BargainController.provider = {
    //  检查状态
    checkStatus: function (options) {
        options.url = options.url || BargainController.api.check;
        options.type = options.type || "POST";
        options.data = "num=1&geid=" + options.data.goodsEntityId;
        BargainController.methods.request(options);
    },
    //  创建活动
    createActivity: function (options) {
        options.url = options.url || BargainController.api.createActivity;
        options.type = options.type || "POST";
        options.data = "num=1&geid=" + options.data.goodsEntityId;
        BargainController.methods.request(options);
    }
};

//  用户控制器
function UserController() {

}

UserController.globalOptions = PublicController.globalOptions;
UserController.methods = PublicController.methods;
//  更新抽奖手机
UserController.methods.updateLotteryMobileNumber = function (options) {
    var self = this
        , delay = 1.5
        , _success = options.success
    ;

    options.success = function (apiData) {
        var handler = {
            1000: function (apiData) {
                art.dialog.tips(apiData.msg, delay);
            },
            1003: function (apiData) {
                location.href = "/mobi/cn/login.html";
            }
        };

        if (handler[apiData.status]) {
            handler[apiData.status].call(self, apiData);
        } else {
            art.dialog.tips(apiData.msg, delay);
        }
        _success(apiData);
    };


    return UserController.provider.updateLotteryMobileNumber(options);
};

//  获取新奖励
UserController.methods.getNewReward = function (options) {
    var _success = options.success;

    options.data.noticeType = 1;
    options.data.giveType = 0;
    options.data.isRead = 0;
    options.data.pageNum = options.data.pageNum || 100;
    options.success = function (apiData) {
        var delay = 1.5
            , handler = {
            1000: function (apiData) {
                _success(apiData);
                if (apiData.data.length > 0) {
                    UserController.provider.updateNoticeMessage({
                        data: {
                            noticeId: apiData.data.map(function (item) {
                                return item.id;
                            })
                        },
                        success: function (apiData) {

                        },
                        error: function (err) {
                            alert("更新消息状态失败：" + JSON.stringify(err));
                        }
                    });
                }
            },
            //  接口异常
            1001: function (apiData) {
                art.dialog.tips(apiData.msg, delay);
            },
            //  用户未登录
            1002: function (apiData) {
                //  不处理
                /*art.dialog.tips("请先登录", delay);
                setTimeout(function () {
                    location.href = "/mobi/cn/login.html";
                }, delay * 1000);*/
            },
            //  缺少参数
            1003: function (apiData) {
                art.dialog.tips(apiData.msg, delay);
            }
        };

        handler[apiData.status](apiData);
    };
    options.error = function (err) {
        alert("获取未读通知失败：" + JSON.stringify(err));
    };
    UserController.provider.getNoticeMessage(options);
};

//  领取奖品
UserController.methods.lotteryPrizeExchangeById = function (id, drawType, options) {
    var delay = 1.5
        ,
        resultInstance
        , showResult = function (options) {
            if (!resultInstance) {
                resultInstance = new Modal();
            }
            resultInstance.init({
                id: "activity-wheel-result",
                containerClass: "activity-wheel-result",
                title: options.title,
                buttons: options.buttons,
                callback: options.callback
            });
            resultInstance.show({
                content: options.content
            });
        },
        hideResult = function () {
            resultInstance.hide();
        }
    ;

    UserController.provider.lotteryPrizeExchange({
        data: {
            id: id,
            type: drawType
        },
        success: function (apiData) {
            var handler = {
                1000: function (apiData) {
                    options.success(apiData);
                },
                //  接口异常
                1001: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                },
                //  该功能未开启
                1002: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                },
                //  需要登录
                1003: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                    setTimeout(function () {
                        location.href = "/mobi/cn/login.html";
                    }, delay * 1000);
                },
                //  缺少参数
                1004: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                },
                //  奖项已领取
                1005: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                },
                //  找不到指定的抽奖活动
                1006: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                },
                //  请在微信中兑换商品
                1007: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                },
                //  需要授权，返回授权页面跳转
                1008: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                    location.href = apiData.redirectUrl;
                },
                //  未关注公众号
                1009: function (apiData) {
                    showResult({
                        title: "请先关注我们",
                        content: '<img class="img" src="' + apiData.imgUrl + '" alt="公众号二维码" /><p class="text">长按二维码关注</p>'
                        /*, buttons: [
                            {
                                title: "暂不",
                                className: "btn-positive"
                            }
                        ],
                        callback: function (index) {
                            hideResult();
                        }*/
                    });
                },
                //  领取失败
                1010: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                },
                //  找不到指定的抽奖记录
                1011: function (apiData) {
                    art.dialog.tips(apiData.msg, delay);
                }
            };

            if (!handler[apiData.status]) {
                return alert(" 无法处理的状态");
            }
            handler[apiData.status].call(this, apiData);
        },
        error: options.error
    });
};

//  接口地址
UserController.apiPrevFix = "";  //  代理
//  接口
UserController.api = {
    noticeMessage: UserController.apiPrevFix + "/user/notice"  //  获取通知消息
    , updateNoticeMessage: UserController.apiPrevFix + "/user/notice/read"  //  更新通知消息
    , lotteryRecord: UserController.apiPrevFix + "/user/lottery/record"  //  获取抽奖记录
    , lotteryPrizeExchange: UserController.apiPrevFix + "/user/lottery/prize/exchange"  //  领取大转盘奖品
    , directPrizeExchange: UserController.apiPrevFix + "/user/draw/prize/exchange"  //  领取一物一码奖品
    , lotteryMobileNumber: UserController.apiPrevFix + "/user/lottery/prize/mobileSubmit"  //  绑定抽奖手机
};

//  数据产出
UserController.provider = {
    //  获取通知信息
    getNoticeMessage: function (options) {
        options.url = options.url || UserController.api.noticeMessage;
        options.type = options.type || "POST";
        options.data = "noticeType=" + options.data.noticeType + "&pageIndex=" + options.data.pageIndex + "&giveType=" + options.data.giveType + "&isRead=" + options.data.isRead + "&pageCount=" + (options.data.pageCount || UserController.globalOptions.pageSize);
        UserController.methods.request(options);
    },
    //  更新消息状态
    updateNoticeMessage: function (options) {
        options.url = options.url || UserController.api.updateNoticeMessage;
        options.type = options.type || "POST";
        options.data = "noticeId=" + options.data.noticeId.join(",");
        UserController.methods.request(options);
    },
    //  获取抽奖记录
    getLotteryRecord: function (options) {
        options.url = options.url || UserController.api.lotteryRecord;
        options.type = options.type || "GET";
        options.data = "type=" + options.data.type + "&pageNum=" + options.data.pageNum + "&pageSize=" + (options.data.pageSize || UserController.globalOptions.pageSize);
        UserController.methods.request(options);
    },
    // 领取奖品
    lotteryPrizeExchange: function (options) {
        options.url = options.url || (options.data.type === 0 ? UserController.api.lotteryPrizeExchange : UserController.api.directPrizeExchange);
        options.type = options.type || "POST";
        options.data = "id=" + options.data.id;
        UserController.methods.request(options);
    },
    // 更新抽奖手机
    updateLotteryMobileNumber: function (options) {
        options.url = options.url || UserController.api.lotteryMobileNumber;
        options.type = options.type || "POST";
        options.data = "id=" + options.data.id + "&mobile=" + options.data.mobileNumber;
        UserController.methods.request(options);
    }
};

UserController.vue = {
    //  我的奖品
    prizeList: function (options) {
        var computed = {
                canRecordLoad: function () {
                    return this.status === 0 || this.status === 2;
                }
            }
            , data = {
                delay: 1.5,
                status: 0,
                imagePlaceHolder: "",
                currentPageNum: 1,
                exchangeStatus: 0,
                prizeList: []
            },
            methods = {
                exchangeByIndex: function (index, luckDrawType) {
                    var vue = this;

                    return UserController.methods.lotteryPrizeExchangeById(this.prizeList[index].id, luckDrawType, {
                        success: function (apiData) {
                            if (apiData.status === 1000) {
                                vue.prizeList[index].isExchanged = true;
                                art.dialog.tips(apiData.msg, vue.delay);
                            }
                        },
                        error: function (err) {
                            alert("领取奖品出错：" + JSON.stringify(err));
                        }
                    });
                },
                getLotteryRecord: function (callback) {
                    var vue = this;

                    if (!this.canRecordLoad) {
                        return false;
                    }
                    callback = callback || function () {

                    };
                    this.status !== 0 && ++this.currentPageNum;
                    this.status = 1;
                    UserController.provider.getLotteryRecord({
                        data: {
                            type: 1,
                            pageNum: this.currentPageNum
                        },
                        success: function (apiData) {
                            var handler = {
                                1000: function (apiData) {
                                    var vue = this;

                                    vue.prizeList = vue.prizeList.concat(apiData.data.map(function (item, index) {
                                        var length = vue.prizeList.length
                                            , _icon = item.icon
                                        ;

                                        PublicController.methods.loader.img([_icon], function (arr) {
                                            arr[0] ? vue.prizeList[index + length].icon = _icon : "";
                                        });
                                        item.icon = vue.imagePlaceHolder;

                                        return item;
                                    }));
                                    vue.status = apiData.data.length ? 2 : 3;
                                },
                                1001: function (apiData) {
                                    art.dialog.tips("请先登录", this.delay);
                                    location.href = (PublicController.methods.isMobileOrPad() ? "/mobi/cn/login.html" : "/cn/login/web.html") + "?redirect=" + location.href;
                                },
                                1002: function (apiData) {
                                    art.dialog.tips(apiData.msg, this.delay);
                                    vue.status = 4;
                                },
                                1003: function (apiData) {
                                    art.dialog.tips(apiData.msg, this.delay);
                                    vue.status = 4;
                                },
                                1004: function (apiData) {
                                    art.dialog.tips(apiData.msg, this.delay);
                                    vue.status = 4;
                                },
                                1005: function (apiData) {
                                    vue.status = 3;
                                }
                            };

                            if (!handler[apiData.status]) {
                                return alert(" 无法处理的状态");
                            }
                            handler[apiData.status].call(vue, apiData);
                            callback(apiData);
                        },
                        error: function (err) {
                            vue.status = 4;
                            alert("获取记录出错：" + JSON.stringify(err));
                        }
                    });
                },
                typeClickHandler: function (type) {
                    var link;

                    switch (type) {
                        case 1:
                            link = PublicController.route.getPathByRouteName(PublicController.route.link.user.couponList);
                            break;
                        default:
                            break;
                    }

                    if (link) {
                        location.href = link;
                    }

                    return false;
                },
                typeToText: function (type) {
                    var str;

                    switch (type) {
                        case 0:
                            str = "无";
                            break;
                        case 1:
                            str = "抵用券";
                            break;
                        case 2:
                            str = "微信红包";
                            break;
                        case 3:
                            str = "积分";
                            break;
                        case 4:
                            str = "其他";
                            break;
                    }

                    return str;
                }
            }
        ;

        options.data = PublicController.methods.objectAssign(data, options.data);
        options.computed = PublicController.methods.objectAssign(computed, options.computed);
        options.methods = PublicController.methods.objectAssign(methods, options.methods);

        return new Vue({
            el: options.el,
            data: options.data,
            computed: options.computed,
            methods: options.methods,
            created: function () {
                var vue = this;

                this.getLotteryRecord(function () {
                    options.created.call(vue);
                });
            },
            mounted: function () {
                options.mounted.call(this);
            }
        });
    }
};
if (typeof SalesOutletController !== "function") {
    //  门店控制
    function SalesOutletController() {

    }
}

//  接口
SalesOutletController.api = {
    salesOutletList: "/salesOutlets/get" //  获取门店信息列表
};

SalesOutletController.globalOptions = {
    pageSize: PublicController.globalOptions.pageSize
};

SalesOutletController.adapter = {
    salesOutletList: function (data) {
        return data;
    }
};

SalesOutletController.provider = {
    //  获取门店列表
    getSalesOutletList: function (options) {
        var _success = options.success;

        options.url = (options.url || SalesOutletController.api.salesOutletList) + "?pageIndex=" + (options.data.pageNum || 1) + "&pageSize=" + (options.data.pageSize || SalesOutletController.globalOptions.pageSize) + "&longitude=" + (options.data.longitude || "") + "&latitude=" + (options.data.latitude || "");
        options.type = options.type || "GET";
        options.data = "";
        options.success = function (apiData) {
            apiData.data = SalesOutletController.adapter.salesOutletList(apiData.data);
            _success(apiData);
        };
        PublicController.methods.request(options);
    }
};

SalesOutletController.vueState = {
    list: function () {
        var data = {
                salesOutlet: {
                    pageNum: 0, //  门店列表页码
                    selectedId: 0, //  选择的门店标识
                    isShowList: false,  //  是否显示选择列表
                    status: 0, //   [0:未开始,1:正在加载,2:部分加载完成,3:加载出错,4:全部加载完成]
                    list: []    //  门店列表信息
                },
                location: {
                    status: 0,
                    longitude: "",
                    latitude: ""
                }    //  位置信息
            }
            , computed = {
                salesOutletSelectedName: function () {
                    var id = this.salesOutlet.selectedId;

                    return id === 0 ? "" : this.salesOutlet.list.find(function (item) {
                        return item.Id === id;
                    }).OutletsTitle;
                }
            }
            , methods = {
                //  阻止穿透事件
                preventEvent: function (event) {
                    event.stopPropagation();
                },
                //  处理距离显示文本
                distanceText: function (distance) {
                    var text;

                    switch (true) {
                        case distance === -1:
                            text = "未知";
                            break;
                        case distance >= 1:
                            text = distance + "km";
                            break;
                        // case distance < 1:
                        default:
                            text = distance * 1000 + "m";
                            break;
                    }

                    return text;
                },
                //  显示地图
                showMapHandler: function () {
                    salesOutletsLocation(this.salesOutlet.selectedId);
                },
                //  门店列表滚动事件
                salesOutletListOnScroll: function (event) {
                    var target = event.target;

                    if (target.scrollHeight - target.clientHeight <= target.scrollTop && this.salesOutlet.status === 2) {
                        this.getSalesOutletList();
                    }
                },
                //  选择门店
                setSalesOutletId: function (id) {
                    this.salesOutlet.selectedId = id;
                    this.switchSalesOutletList(false);
                },
                //  切换门店列表显示与否
                switchSalesOutletList: function (toShow) {
                    if (toShow && !this.salesOutlet.list.length) {
                        this.getSalesOutletList();
                    }
                    this.salesOutlet.isShowList = toShow;
                },
                //  获取下一页门店列表
                getSalesOutletList: function () {
                    var vue = this
                        , getLocation = function (callback) {
                            if (vue.location.status === 0 || vue.location.status === 5) {
                                if (MainController.inWechat()) {
                                    vue.location.status = 2;
                                    PublicController.methods.getLocationByWeChat({
                                        success: function (apiData) {
                                            vue.location.status = 4;
                                            vue.location.latitude = apiData.latitude;
                                            vue.location.longitude = apiData.longitude;
                                            callback();
                                        },
                                        error: function (err) {
                                            vue.location.status = 5;
                                            callback();
                                        }
                                    });
                                } else {
                                    vue.location.status = 3;
                                    callback();
                                }
                            } else {
                                callback();
                            }
                        }
                    ;

                    if (vue.salesOutlet.status === 0 || vue.salesOutlet.status === 2) {
                        getLocation(function () {
                            vue.salesOutlet.pageNum = vue.salesOutlet.pageNum + 1;
                            vue.salesOutlet.status = 1;
                            SalesOutletController.provider.getSalesOutletList({
                                data: {
                                    pageNum: vue.salesOutlet.pageNum,
                                    latitude: vue.location.latitude
                                    , longitude: vue.location.longitude
                                },
                                success: function (apiData) {
                                    var statusHandler = {
                                        //  成功
                                        1000: function (apiData) {
                                            if (apiData.salesOutletsList.length === 0) {
                                                vue.salesOutlet.status = 4;
                                            } else {
                                                vue.salesOutlet.list.push.apply(vue.salesOutlet.list, apiData.salesOutletsList);
                                                vue.salesOutlet.status = 2;
                                            }
                                        },
                                        //  出现异常
                                        1001: function (apiData) {
                                            vue.salesOutlet.status = 3;
                                            alert(apiData.msg);
                                        },
                                        //  缺少参数
                                        1002: function (apiData) {
                                            vue.salesOutlet.status = 3;
                                            alert(apiData.msg);
                                        },
                                        //  功能未启用
                                        1003: function (apiData) {
                                            vue.salesOutlet.status = 3;
                                            alert(apiData.msg);
                                        }
                                    };

                                    statusHandler[apiData.status].call(vue, apiData);
                                },
                                error: function (err) {
                                    vue.salesOutlet.status = 3;
                                    alert("获取门店信息列表错误：" + JSON.stringify(err));
                                }
                            });
                        });
                    }
                }
            }
        ;

        return {
            data: data,
            computed: computed,
            methods: methods
        };
    }
};

if (typeof OrderController !== "function") {
    function OrderController() {

    }
}

OrderController.api = {
    checkStatus: "/ashx/refresh_status.ashx"
};

OrderController.provider = {
    //  检查支付状态
    checkPayStatusByOrderNumber: function (options) {
        return PublicController.methods.request({
            type: "POST",
            url: OrderController.api.checkStatus,
            data: "orderNo=" + options.data.orderNumber,
            success: options.success,
            error: options.error
        });
    }
};

//  vue实例结构
OrderController.vue = {
    //  拼团
    spellToOrder: function (options) {
        var salesOutletListVueState = SalesOutletController.vueState.list()
            , data = {}
            , computed = {}
            , methods = {}
        ;

        options.data = PublicController.methods.objectAssign(data, salesOutletListVueState.data, options.data);
        options.computed = PublicController.methods.objectAssign(computed, salesOutletListVueState.computed, options.computed);
        options.methods = PublicController.methods.objectAssign(methods, salesOutletListVueState.methods, options.methods);

        return new Vue({
            el: options.el,
            data: options.data,
            computed: options.computed,
            methods: options.methods,
            created: function () {
                var vue = this;

                //  拼团订单提交
                $("#spellOrderSumitForm").submit(function () {
                    var delay = 1.5;

                    // 只要应付金额不为零即可（商品价格和运费其中之一不为零）
                    //if (parseInt($("#sSumTotal").text().split("¥")[1]) == 0) {
                    if (document.getElementById("hdGoodsGeid").value == "") {
                        art.dialog.tips("购物车没有加入任何商品！", delay);
                        setTimeout("window.location.href='/mobi/index.html'", delay * 1000);
                        return false;
                    }
                    if ($("input[name='receiptWay']:checked").val() == 0) {
                        var userInfo = "",  //  订单信息
                            deliveryDate = "",  //  配送开始和结束日期
                            deliveryTime = "",  //  配送的时间选择
                            isDelivery = "",    //  是否配送,
                            da = "";    //  用户地址

                        if ($("input[name='userAddress']:checked").val() == undefined || $("input[name='userAddress']:checked").val() == null) {
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
                            if ($("#txtZipcode").val() !== "" && !formatCheck($("#txtZipcode").val())) {
                                $("#txtZipcode").focus();
                                art.dialog.tips("邮政编码格式不符！", delay);
                                return false;
                            }
                            if ($("#txtTelephone").val() == "" && $("#txtMobile").val() == "") {
                                $("#txtTelephone").focus();
                                art.dialog.tips("联系电话和手机号码至少填写一项！", delay);
                                return false;
                            }
                        }
                        if ($("input[name='deliveryWay']:checked").val() == null) {
                            $("input[name='deliveryWay']")[0].focus();
                            art.dialog.tips("请选择配送方式！", delay);
                            return false;
                        }
                        if (document.getElementById('OrderTime')) {
                            if (document.getElementById("txtStartDate").value == "") {
                                art.dialog.tips("请输入开始日期！", delay);
                                $("#txtStartDate").select();
                                return false;
                            }
                            if (document.getElementById("txtEndDate").value == "") {
                                art.dialog.tips("请输入结束日期！", delay);
                                $("#txtEndDate").select();
                                return false;
                            }
                        }
                        if ($("#txtStartDate").length > 0 && $("#txtEndDate").length > 0) {
                            var d1Arr = document.getElementById("txtStartDate").value.split('-');
                            var d2Arr = document.getElementById("txtEndDate").value.split('-');
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
                                deliveryDate = $("#txtStartDate").val() + "至" + $("#txtEndDate").val();
                            }
                        }
                        if ($("#hdDeliveryWay").val() == "1") {
                            if ($("input[name='cbDeliveryTime']:checked").val() != null) {
                                $("input[name='cbDeliveryTime']").each(function () {
                                    if ($(this).attr("checked")) {
                                        deliveryTime += "," + $(this).val();
                                        isDelivery += $(this).attr("checked") ? ",1" : ",0";
                                    }
                                });
                                deliveryTime = deliveryTime.replace(/^,/, '');
                                isDelivery = isDelivery.replace(/^,/, '');
                            }
                        }
                        if ($("#txtaPostscript").val().length > $("#txtaPostscript").attr('maxlength')) {
                            $("#txtaPostscript").focus();
                            art.dialog.tips("附言不能超过" + $("#txtaPostscript").attr('maxlength') + "个字符！", delay);
                            return false;
                        } else if ($("#cbInvoice").is(":checked") === true && $("#txtaPostscript").val().length === 0) {
                            $("#txtaPostscript").focus();
                            art.dialog.tips("请在附言中注明您的发票抬头！", delay);
                            return false;
                        }
                        if ($("input[name='userAddress']:checked").size() == 0) {
                            art.dialog.tips("请先填写收货地址！", delay);
                            setTimeout("window.location.href='/mobi/cn/member/shipping/address_0.html'", 2000);
                            return false;
                        } else {
                            da = "&userAddress=" + $("input[name='userAddress']:checked").val();
                        }

                        art.dialog.tips("订单提交中，请耐心等候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/spell_order_info.ashx",
                            data: "deliveryWay=" + $("input[name='deliveryWay']:checked").val() + da + "&postscript=" + encodeURI($("#txtaPostscript").val()) + "&goodsSum=" + $("#hdGoodsSum").val() + "&deliveryDate=" + deliveryDate + "&deliveryTime=" + deliveryTime + "&isDelivery=" + isDelivery + ($("#hdAnonymous").length > 0 ? "&Anonymous=1" : "") + "&isInvoice=" + $("#cbInvoice").is(':checked') + "&geid=" + $("#hdGoodsGeid").val(),
                            success: spellOrderSubmitSuccessCB
                        });
                    } else {
                        if (vue.salesOutlet.selectedId === 0) {
                            vue.switchSalesOutletList(true);
                            art.dialog.tips("请选择自提分店！", delay);

                            return false;
                        }
                        if ($("#txtSOConsignee").val() == "") {
                            $("#txtSOConsignee").focus();
                            art.dialog.tips("请输入联系人！", delay);
                            return false;
                        }
                        if ($("#txtSOMobile").val() == "" || !/^(((13[0-9]{1})|(15[0-9]{1})|(18[0-9]{1})|(177))+\d{8})$/.test($("#txtSOMobile").val())) {
                            $("#txtSOMobile").focus();
                            art.dialog.tips("请输入正确的手机号码！", delay);
                            return false;
                        }
                        if ($("#txtaPostscript").val().length > 200) {
                            $("#txtaPostscript").select();
                            art.dialog.tips("附言不能超过200个字符！", delay);
                            return false;
                        }
                        if ($("#hdGoodsGeid").val() == "") {
                            art.dialog.tips("购物车没有加入任何商品！", delay);
                            var lurl = "window.location.href='/mobi/cn/index.html'";
                            setTimeout(lurl, delay * 1000);
                            return false;
                        }
                        art.dialog.tips("订单提交中，请耐心等候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", 30);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/spell_order_info.ashx",
                            data: "salesOutlets=" + vue.salesOutlet.selectedId + "&sOConsignee=" + encodeURI($("#txtSOConsignee").val()) + "&sOMobile=" + $("#txtSOMobile").val() + "&postscript=" + encodeURI($("#txtaPostscript").val()) + "&goodsSum=" + $("#hdGoodsSum").val() + ($("#hdAnonymous").length > 0 ? "&Anonymous=1" : "") + "&isInvoice=" + $("#cbInvoice").is(':checked') + "&geid=" + $("#hdGoodsGeid").val() + "&deliveryDate=" + deliveryDate + "&deliveryTime=" + deliveryTime + "&isDelivery=" + isDelivery,
                            success: spellOrderSubmitSuccessCB
                        });
                    }
                    return false;
                });
            }
        });
    },
    //  秒杀
    secKillToOrder: function (options) {
        var salesOutletListVueState = SalesOutletController.vueState.list()
            , data = {}
            , computed = {}
            , methods = {}
        ;

        options.data = PublicController.methods.objectAssign(data, salesOutletListVueState.data, options.data);
        options.computed = PublicController.methods.objectAssign(computed, salesOutletListVueState.computed, options.computed);
        options.methods = PublicController.methods.objectAssign(methods, salesOutletListVueState.methods, options.methods);

        return new Vue({
            el: options.el,
            data: options.data,
            computed: options.computed,
            methods: options.methods,
            created: function () {
                var vue = this;

                //  秒杀订单提交
                $("#seckillOrderSumitForm").submit(function () {
                    var delay = 1.5
                        , waitDelay = 30
                    ;

                    // 只要应付金额不为零即可（商品价格和运费其中之一不为零）
                    //if (parseInt($("#sSumTotal").text().split("¥")[1]) == 0) {
                    if (document.getElementById("hdGoodsGeid").value == "") {
                        art.dialog.tips("购物车没有加入任何商品！", delay);
                        setTimeout("window.location.href='/mobi/index.html'", delay * 1000);
                        return false;
                    }
                    if ($("input[name='receiptWay']:checked").val() == 0) {
                        var userInfo = "",  //  订单信息
                            deliveryDate = "",  //  配送开始和结束日期
                            deliveryTime = "",  //  配送的时间选择
                            isDelivery = "",    //  是否配送,
                            da = "";    //  用户地址

                        if ($("input[name='userAddress']:checked").val() == undefined || $("input[name='userAddress']:checked").val() == null) {
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
                        if (document.getElementById('OrderTime')) {
                            if (document.getElementById("txtStartDate").value == "") {
                                art.dialog.tips("请输入开始日期！", delay);
                                $("#txtStartDate").select();
                                return false;
                            }
                            if (document.getElementById("txtEndDate").value == "") {
                                art.dialog.tips("请输入结束日期！", delay);
                                $("#txtEndDate").select();
                                return false;
                            }
                        }
                        if ($("#txtStartDate").length > 0 && $("#txtEndDate").length > 0) {
                            var d1Arr = document.getElementById("txtStartDate").value.split('-');
                            var d2Arr = document.getElementById("txtEndDate").value.split('-');
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
                                deliveryDate = $("#txtStartDate").val() + "至" + $("#txtEndDate").val();
                            }
                        }
                        if ($("#hdDeliveryWay").val() == "1") {
                            if ($("input[name='cbDeliveryTime']:checked").val() != null) {
                                $("input[name='cbDeliveryTime']").each(function () {
                                    if ($(this).attr("checked")) {
                                        deliveryTime += "," + $(this).val();
                                        isDelivery += $(this).attr("checked") ? ",1" : ",0";
                                    }
                                });
                                deliveryTime = deliveryTime.replace(/^,/, '');
                                isDelivery = isDelivery.replace(/^,/, '');
                            }
                        }
                        if ($("#txtaPostscript").val().length > $("#txtaPostscript").attr('maxlength')) {
                            $("#txtaPostscript").focus();
                            art.dialog.tips("附言不能超过" + $("#txtaPostscript").attr('maxlength') + "个字符！", delay);
                            return false;
                        } else if ($("#cbInvoice").is(":checked") === true && $("#txtaPostscript").val().length === 0) {
                            $("#txtaPostscript").focus();
                            art.dialog.tips("请在附言中注明您的发票抬头！", delay);
                            return false;
                        }
                        if ($("input[name='userAddress']:checked").size() == 0) {
                            art.dialog.tips("请先填写收货地址！", delay);
                            setTimeout("window.location.href='/mobi/cn/member/shipping/address_0.html'", delay * 1000);
                            return false;
                        } else {
                            da = "&userAddress=" + $("input[name='userAddress']:checked").val();
                        }

                        art.dialog.tips("订单提交中，请耐心等候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", waitDelay);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/seckill_order_info.ashx",
                            data: "deliveryWay=" + $("input[name='deliveryWay']:checked").val() + da + "&postscript=" + encodeURI($("#txtaPostscript").val()) + "&goodsSum=" + $("#hdGoodsSum").val() + "&deliveryDate=" + deliveryDate + "&deliveryTime=" + deliveryTime + "&isDelivery=" + isDelivery + ($("#hdAnonymous").length > 0 ? "&Anonymous=1" : "") + "&isInvoice=" + $("#cbInvoice").is(':checked') + "&coupon=" + $("#hdCoupon").val() + "&geid=" + $("#hdGoodsGeid").val(),
                            success: seckillOrderSubmitSuccessCB
                        });
                    } else {
                        if (vue.salesOutlet.selectedId === 0) {
                            vue.switchSalesOutletList(true);
                            art.dialog.tips("请选择自提分店！", delay);

                            return false;
                        }
                        if ($("#txtSOConsignee").val() == "") {
                            $("#txtSOConsignee").focus();
                            art.dialog.tips("请输入联系人！", delay);
                            return false;
                        }
                        if ($("#txtSOMobile").val() == "" || !/^(((13[0-9]{1})|(15[0-9]{1})|(17[0-9]{1})|(18[0-9]{1}))+\d{8})$/.test($("#txtSOMobile").val())) {
                            $("#txtSOMobile").focus();
                            art.dialog.tips("请输入正确的手机号码！", delay);
                            return false;
                        }
                        if ($("#txtaPostscript").val().length > 200) {
                            $("#txtaPostscript").select();
                            art.dialog.tips("附言不能超过200个字符！", delay);
                            return false;
                        }
                        if ($("#hdGoodsGeid").val() == "") {
                            art.dialog.tips("购物车没有加入任何商品！", delay);
                            var lurl = "location.href='/mobi/cn/index.html'";
                            setTimeout(lurl, delay * 1000);
                            return false;
                        }
                        art.dialog.tips("订单提交中，请耐心等候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", waitDelay);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/seckill_order_info.ashx",
                            data: "salesOutlets=" + vue.salesOutlet.selectedId + "&sOConsignee=" + encodeURI($("#txtSOConsignee").val()) + "&sOMobile=" + $("#txtSOMobile").val() + "&postscript=" + encodeURI($("#txtaPostscript").val()) + "&goodsSum=" + $("#hdGoodsSum").val() + ($("#hdAnonymous").length > 0 ? "&Anonymous=1" : "") + "&isInvoice=" + $("#cbInvoice").is(':checked') + "&coupon=" + $("#hdCoupon").val() + "&geid=" + $("#hdGoodsGeid").val() + "&deliveryDate=" + deliveryDate + "&deliveryTime=" + deliveryTime + "&isDelivery=" + isDelivery,
                            success: seckillOrderSubmitSuccessCB
                        });
                    }
                    return false;
                });
            }
        });
    },
    //  商品转订单
    goodsToOrder: function (options) {
        var salesOutletListVueState = SalesOutletController.vueState.list()
            , data = {}
            , computed = {}
            , methods = {}
        ;

        options.data = PublicController.methods.objectAssign(data, salesOutletListVueState.data, options.data);
        options.computed = PublicController.methods.objectAssign(computed, salesOutletListVueState.computed, options.computed);
        options.methods = PublicController.methods.objectAssign(methods, salesOutletListVueState.methods, options.methods);

        return new Vue({
            el: options.el,
            data: options.data,
            computed: options.computed,
            methods: options.methods,
            watch: {
                salesOutlet: function (oldVal, newVal) {
                    $("body")[newVal.isShow ? "addClass" : "removeClass"]("mainHidden");
                }
            },
            created: function () {
                var vue = this;

                //  普通订单提交
                $("#orderSumitForm").submit(function () {
                    var delay = 1.5
                        , waitDelay = 30
                    ;

                    // 只要应付金额不为零即可（商品价格和运费其中之一不为零）
                    //if (parseInt($("#sSumTotal").text().split("¥")[1]) == 0) {
                    if (document.getElementById("hdGoodsGeid").value == "") {
                        art.dialog.tips("购物车没有加入任何商品！", delay);
                        setTimeout("window.location.href='/mobi/index.html'", delay * 1000);
                        return false;
                    }
                    if ($("input[name='receiptWay']:checked").val() == 0) {
                        var userInfo = "",  //  订单信息
                            deliveryDate = "",  //  配送开始和结束日期
                            deliveryTime = "",  //  配送的时间选择
                            isDelivery = "",    //  是否配送
                            da = "";    //  用户地址

                        if ($("input[name='userAddress']:checked").val() == undefined || $("input[name='userAddress']:checked").val() == null) {
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
                        if (document.getElementById('OrderTime')) {
                            if (document.getElementById("txtStartDate").value == "") {
                                art.dialog.tips("请输入开始日期！", delay);
                                $("#txtStartDate").select();
                                return false;
                            }
                            if (document.getElementById("txtEndDate").value == "") {
                                art.dialog.tips("请输入结束日期！", delay);
                                $("#txtEndDate").select();
                                return false;
                            }
                        }
                        if ($("#txtStartDate").length > 0 && $("#txtEndDate").length > 0) {
                            var d1Arr = document.getElementById("txtStartDate").value.split('-');
                            var d2Arr = document.getElementById("txtEndDate").value.split('-');
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
                                deliveryDate = $("#txtStartDate").val() + "至" + $("#txtEndDate").val();
                            }
                        }
                        if ($("#hdDeliveryWay").val() == "1") {
                            if ($("input[name='cbDeliveryTime']:checked").val() != null) {
                                $("input[name='cbDeliveryTime']").each(function () {
                                    if ($(this).attr("checked")) {
                                        deliveryTime += "," + $(this).val();
                                        isDelivery += $(this).attr("checked") ? ",1" : ",0";
                                    }
                                });
                                deliveryTime = deliveryTime.replace(/^,/, '');
                                isDelivery = isDelivery.replace(/^,/, '');
                            }
                        }
                        if ($("#salesOutletsDistribution option:selected").val() == "0") {
                            $("#salesOutletsDistribution").focus();
                            art.dialog.tips("请选择分配门店！", delay);
                            return false;
                        }
                        if ($("#txtaPostscript").val().length > $("#txtaPostscript").attr('maxlength')) {
                            $("#txtaPostscript").focus();
                            art.dialog.tips("附言不能超过" + $("#txtaPostscript").attr('maxlength') + "个字符！", delay);
                            return false;
                        } else if ($("#cbInvoice").is(":checked") === true && $("#txtaPostscript").val().length === 0) {
                            $("#txtaPostscript").focus();
                            art.dialog.tips("请在附言中注明您的发票抬头！", delay);
                            return false;
                        }
                        if ($("input[name='userAddress']:checked").size() == 0) {
                            art.dialog.tips("请先填写收货地址！", delay);
                            setTimeout("window.location.href='/mobi/cn/member/shipping/address_0.html'", delay * 1000);
                            return false;
                        } else {
                            da = "&userAddress=" + $("input[name='userAddress']:checked").val();
                        }

                        art.dialog.tips("订单提交中，请耐心等候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", waitDelay);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/order_info.ashx",
                            data: "deliveryWay=" + $("input[name='deliveryWay']:checked").val() + da + "&postscript=" + encodeURI($("#txtaPostscript").val()) + "&goodsSum=" + $("#hdGoodsSum").val() + "&deliveryDate=" + deliveryDate + "&deliveryTime=" + deliveryTime + "&isDelivery=" + isDelivery + ($("#hdAnonymous").length > 0 ? "&Anonymous=1" : "") + "&isInvoice=" + $("#cbInvoice").is(':checked') + "&coupon=" + $("#hdCoupon").val() + "&geid=" + $("#hdGoodsGeid").val() + "&buyNowAmount=" + $("#hdBuyNowAmount").val() + "&salesOutletsDistribution=" + $("#salesOutletsDistribution option:selected").val(),
                            success: orderSubmitSuccessCB
                        });
                    } else {
                        if (vue.salesOutlet.selectedId === 0) {
                            vue.switchSalesOutletList(true);
                            art.dialog.tips("请选择自提分店！", delay);

                            return false;
                        }
                        if ($("#txtSOConsignee").val() == "") {
                            $("#txtSOConsignee").focus();
                            art.dialog.tips("请输入联系人！", delay);
                            return false;
                        }
                        if ($("#txtSOMobile").val() == "" || !/^(((13[0-9]{1})|(15[0-9]{1})|(17[0-9]{1})|(18[0-9]{1}))+\d{8})$/.test($("#txtSOMobile").val())) {
                            $("#txtSOMobile").focus();
                            art.dialog.tips("请输入正确的手机号码！", delay);
                            return false;
                        }
                        if ($("#txtaPostscript").val().length > 200) {
                            $("#txtaPostscript").select();
                            art.dialog.tips("附言不能超过200个字符！", delay);
                            return false;
                        }
                        if ($("#hdGoodsGeid").val() == "") {
                            art.dialog.tips("购物车没有加入任何商品！", delay);
                            var lurl = "window.location.href='/mobi/cn/index.html'";
                            setTimeout(lurl, delay * 1000);
                            return false;
                        }
                        art.dialog.tips("订单提交中，请耐心等候 <img style=\"vertical-align:middle;\" src=\"/images/public/images/o_loading.gif\" />", waitDelay);
                        $.ajax({
                            type: "POST",
                            url: "/ashx/cn/order_info.ashx",
                            data: "salesOutlets=" + vue.salesOutlet.selectedId + "&sOConsignee=" + encodeURI($("#txtSOConsignee").val()) + "&sOMobile=" + $("#txtSOMobile").val() + "&postscript=" + encodeURI($("#txtaPostscript").val()) + "&goodsSum=" + $("#hdGoodsSum").val() + ($("#hdAnonymous").length > 0 ? "&Anonymous=1" : "") + "&isInvoice=" + $("#cbInvoice").is(':checked') + "&coupon=" + $("#hdCoupon").val() + "&geid=" + $("#hdGoodsGeid").val() + "&deliveryDate=" + deliveryDate + "&deliveryTime=" + deliveryTime + "&isDelivery=" + isDelivery + "&buyNowAmount=" + $("#hdBuyNowAmount").val(),
                            success: orderSubmitSuccessCB
                        });
                    }

                    return false;
                });
                options.created.call(vue);
            }
        });
    }
};

function PaymentController() {

}

PaymentController.api = {
    offLineOrderGenerate: "/scanCode/orderSubmit"    //  线下扫码支付提交
};


PaymentController.provider = {
    //  线下订单生成
    offLineOrderGenerate: function (options) {
        PublicController.methods.request({
            type: "POST",
            url: PaymentController.api.offLineOrderGenerate,
            data: "orderTotal=" + options.data.orderTotal + "&soId=" + options.data.storeId,
            success: options.success,
            error: options.error
        });
    }
};

PaymentController.vue = {
    offLinePay: function (options) {
        var salesOutletListVueState = SalesOutletController.vueState.list()
            , data = {
                storeId: "",
                delay: 1.5,
                payType: "",
                output: {
                    value: ""
                },
                input: {
                    dealing: false,    //  是否正在处理输入
                    clickType: "input-button-click",
                    submit: {
                        type: "submit",
                        title: "立即支付"
                    },
                    buttons: [
                        {
                            title: 1
                        }
                        , {
                            title: 2
                        }
                        , {
                            title: 3
                        }
                        , {
                            title: 4
                        }
                        , {
                            title: 5
                        }
                        , {
                            title: 6
                        }
                        , {
                            title: 7
                        }
                        , {
                            title: 8
                        }
                        , {
                            title: 9
                        }
                        , {
                            title: "."
                        }
                        , {
                            title: 0
                        }
                        , {
                            className: "icon-delete-3"
                        }
                    ]
                }
            }
            , computed = {
                inputSubmitDisabled: function () {
                    return !this.payType || Number(this.output.value) <= 0 || this.input.dealing;
                }
            }
            , methods = {
                handler: function (type, data) {
                    var handler = {};

                    handler[this.input.clickType] = function (index) {
                        var value = this.output.value
                            , title = this.input.buttons[index].title
                        ;

                        if (title !== undefined) {
                            if (title === ".") {
                                value = value.replace(title, "") + title;
                            } else {
                                value += title.toString();
                            }
                        } else {
                            value = value.replace(/\d[.]?$/, "");
                        }
                        this.output.value = value;
                    };
                    handler[this.input.submit.type] = function () {
                        var vue = this
                            , orderTotal = Number(this.output.value)
                            , tipText = ""
                        ;

                        if (vue.inputSubmitDisabled) {
                            if (!vue.payType) {
                                tipText = "无效的支付环境";
                            } else if (orderTotal) {
                                tipText = "无效的支付金额";
                            } else if (vue.input.dealing) {
                                tipText = "支付处理中，请稍候";
                            } else {
                                tipText = "无法处理的状态";
                            }
                            art.dialog.tips(tipText, vue.delay);

                            return false;
                        }
                        vue.input.dealing = true;
                        PaymentController.provider.offLineOrderGenerate({
                            data: {
                                orderTotal: orderTotal
                                , storeId: vue.storeId
                            },
                            success: function (apiData) {
                                var handler = {
                                    1000: function (apiData) {
                                        payment(vue.payType, apiData.orderNo, {
                                            hdfMicroMessenger: 0,
                                            hdfWxpay: 2
                                        }, null, orderTotal, function (err) {
                                            alert("支付出错：" + JSON.stringify(err));
                                            vue.input.dealing = false;
                                        });
                                    },
                                    //  接口异常
                                    1001: function (apiData) {
                                        art.dialog.tips(apiData.msg, this.delay);
                                        vue.input.dealing = false;
                                    },
                                    //  支付金额必须大于0
                                    1002: function (apiData) {
                                        art.dialog.tips(apiData.msg, this.delay);
                                        vue.input.dealing = false;
                                    },
                                    //  订单提交失败
                                    1003: function (apiData) {
                                        art.dialog.tips(apiData.msg, this.delay);
                                        vue.input.dealing = false;
                                    }
                                };

                                handler[apiData.status].call(vue, apiData);
                            },
                            error: function (err) {
                                art.dialog.tips("生成订单出错：" + JSON.stringify(err), vue.delay);
                            }
                        });
                    };
                    if (handler[type]) {
                        handler[type].call(this, data);
                    } else {
                        art.dialog.tips("无法处理的事件", this.delay);
                    }
                }
            }
        ;

        options.data = PublicController.methods.objectAssign(data, options.data);
        options.computed = PublicController.methods.objectAssign(computed, options.computed);
        options.methods = PublicController.methods.objectAssign(methods, options.methods);

        return new Vue({
            el: options.el,
            data: options.data,
            computed: options.computed,
            methods: options.methods,
            created: function () {
                var vue = this
                    //  客户端类型
                    , payTypeList = [
                        {
                            title: "支付宝",
                            type: payment.type.aLi,
                            check: MainController.inALiPay
                        },
                        {
                            title: "微信",
                            type: payment.type.weChat,
                            check: MainController.inWechat
                        }
                    ]
                ;

                payTypeList.forEach(function (item) {
                    if (item.check()) {
                        vue.input.submit.title = item.title + "支付";
                        vue.payType = item.type;

                        return false;
                    }
                });
                // vue.payType = payment.type.weChat;   //  测试
                if (!vue.payType) {
                    var splitText = "或者";

                    art.dialog.tips("请在" + payTypeList.reduce(function (prev, item) {
                        return prev + splitText + item.title;
                    }, "").replace(new RegExp("^" + splitText), "") + "中打开", this.delay);
                }
                options.created.call(vue);
            }
        });
    }
};