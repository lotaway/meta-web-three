function sendMessage(message){
    art.dialog({time: 2, content: message,lock: true,fixed:true,opacity:0.1});
}
function btnSendMessage(message){
    art.dialog.alert(message);
}
function customButton(message,btnTitle1,url,btnTitle2){
    art.dialog({
        id: 'testID',
        content: message,
        lock: true,
        fixed:true,
        opacity:0.1,
        button: [
            {
                name: btnTitle1,
                callback: function () {
                    window.location=url;
                    return false;
                },
                focus: true
            },
            {
                name: btnTitle2,
                callback: function () {
                }
            }
        ]
    });
}
function customButtonShoppingCart(message,btnTitle1,sid,btnTitle2){
    art.dialog({
        id: 'testID',
        content: message,
        lock: true,
        fixed:true,
        opacity:0.1,
        button: [
            {
                name: btnTitle1,
                callback: function () {
                    deleteShoppingCart(sid);
                },
                focus: true
            },
            {
                name: btnTitle2,
                callback: function () {
                }
            }
        ]
    });
}

function customButtonShoppingCartGoodsPackage(message,btnTitle1,goodsPackage,btnTitle2){
    art.dialog({
        id: 'testID',
        content: message,
        lock: true,
        fixed:true,
        opacity:0.1,
        button: [
            {
                name: btnTitle1,
                callback: function () {
                    deleteShoppingCartGoodsPackage(goodsPackage);
                },
                focus: true
            },
            {
                name: btnTitle2,
                callback: function () {
                }
            }
        ]
    });
}

function customButtonbgShoppingCart(message, btnTitle1, sid, btnTitle2) {
    art.dialog({
        id: 'testID',
        content: message,
        lock: true,
        fixed: true,
        opacity: 0.1,
        button: [
            {
                name: btnTitle1,
                callback: function () {
                    deletebgShoppingCart(sid);
                },
                focus: true
            },
            {
                name: btnTitle2,
                callback: function () {
                }
            }
        ]
    });
}

function confirmMessage(message,url){
    art.dialog.confirm(message, function(){
        window.location.href=url;
        return true;
    }, function(){
        art.dialog.tips('你取消了操作');
    });
}

function customButtonLogin(message,btnTitle1,url1,btnTitle2,url2){
    art.dialog({
        id: 'testID',
        content: message,
        lock: true,
        fixed:true,
        opacity:0.1,
        button: [
            {
                name: btnTitle1,
                callback: function () {
                    window.location=url1;
                    return false;
                },
                focus: true
            },
            {
                name: btnTitle2,
                callback: function () {
                    window.location=url2;
                    return false;
                }
            }
        ]
    });
}

function customThreeButtonShoppingCart(message,btnTitle1,url1,btnTitle2,url2,btnTitle3,url3){
    art.dialog({
        id: 'testID',
        content: message,
        lock: true,
        fixed:true,
        opacity:0.1,
        button: [
            {
                name: btnTitle1,
                callback: function () {
                    window.location=url1;
                    return false;
                },
                focus: true
            },
            {
                name: btnTitle2,
                callback: function () {
                    window.location=url2;
                    return false;
                }
            },
            {
                name: btnTitle3,
                callback: function () {
                    window.location=url3;
                    return false;
                }
            }
        ]
    });
}

function receiveConfirm(orderNo, rsakey) {
    //alert(rsakey);
    art.artDialog.prompt3(
        "订单收货管理员确认",
        function (inputaccount,pwd) {
            //alert(inputaccount+" "+ pwd + " " + orderNo);
            //setMaxDigits(129);
            //var key = new RSAKeyPair(strPublicKeyExponent, "", strPublicKeyModulus);
            var pwdRtn = encryptedString(rsakey, pwd);
            //alert(pwdRtn);

            $.ajax({
                    //type:"GET",
                    type:"POST",
                    url:"/ashx/mobi/OrderReceiveConfirm.ashx",
                    //global: true,//default:false
                    async: false,//default:true
                    dataType:"text",
                    data: "orderNo=" + orderNo +"&maccount="+inputaccount+ "&pwd=" + pwdRtn,
                    success:function(msg){
                            if(parseInt(msg)==1){
                              location.reload();
                            }
                            else
                            art.dialog.tips(msg,null,1.5);
                    },
		            error:function(){
			            alert('网络超时');
			            //art.dialog.tips("网络超时",null,1.5);
			            //alert("提交超时");
		            }
                });
            },
        ""
        );
}
function SrandomString(len) {
　　len = len || 32;
　　var $chars = 'ABCDEFGHJKMNPQRSTWXYZabcdefhijkmnprstwxyz2345678';    /****默认去掉了容易混淆的字符oOLl,9gq,Vv,Uu,I1****/
　　var maxPos = $chars.length;
　　var pwd = '';
　　for (i = 0; i < len; i++) {
　　　　pwd += $chars.charAt(Math.floor(Math.random() * maxPos));
　　}
　　return pwd;
}

function waitPayResultCrowdfund(message, btnTitle1, UrlTurnTo, btnTitle2) {
    art.dialog({
        id: 'testID',
        content: message,
        lock: true,
        fixed: true,
        opacity: 0.1,
        button: [
            {
                name: btnTitle1,
                callback: function () {
                    top.location.href = UrlTurnTo;
                },
                focus: true
            },
            {
                name: btnTitle2,
                callback: function () {
                }
            }
        ]
    });
}

/* use by DX */
artDialog.prompt3 = function (content, yes, value) {
    value = value || '';
    var input,inputaccount;

    return artDialog({
        id: 'Prompt',
        //zIndex: _zIndex(),
        icon: 'question',
        fixed: true,
        lock: true,
        opacity: .1,
        content: [
			'<div style="margin-bottom:5px;font-size:12px;padding-left: 3px;" class="artDialog_prompt3_title">',
				content,
			'</div>',
			'<div>',
				'<input name="orderconfirmmaccount" placeholder="管理员帐号" class="artDialog_prompt3_input" value="',
					value,
				'" style="width:10em;padding:6px 2px;margin-bottom:3px;" autocomplete="off" type="text" /><br/>',
				'<input name="orderconfirmmpwd" placeholder="管理员密码" class="artDialog_prompt3_input" value="',
					value,
				'" style="width:10em;padding:6px 2px" autocomplete="off" type="password" />',
			'</div>'
        ].join(''),
        init: function () {
            inputaccount = this.DOM.content.find('input[name=\'orderconfirmmaccount\']')[0];
            inputaccount.value = maccountValue;
            input = this.DOM.content.find('input[name=\'orderconfirmmpwd\']')[0];
            input.select();
            input.focus();
        },
        ok: function (here) {
            return yes && yes.call(this,inputaccount.value, input.value, here);
        },
        cancel: true
    });
};

artDialog.promptBargain = function (content, yes, value) {
    value = value || '';
    var input, inputaccount;

    return artDialog({
        id: 'Prompt',
        icon: 'question',
        fixed: true,
        lock: true,
        opacity: .1,
        content: [
			'<div style="margin-bottom:5px;font-size:12px;padding-left: 3px;" class="artDialog_prompt3_title">',
				content,
			'</div>',
			'<div>',
				'<input name="orderconfirmmaccount" placeholder="管理员帐号" class="artDialog_prompt3_input" value="',
					value,
				'" style="width:10em;padding:6px 2px;margin-bottom:3px;" autocomplete="off" type="text" /><br/>',
				'<input name="orderconfirmmpwd" placeholder="管理员密码" class="artDialog_prompt3_input" value="',
					value,
				'" style="width:10em;padding:6px 2px" autocomplete="off" type="password" /><br/>',
			'</div>'
        ].join(''),
        init: function () {
            inputaccount = this.DOM.content.find('input[name=\'orderconfirmmaccount\']')[0];
            inputaccount.value = maccountValue;
            inputaccount.select();
            inputaccount.focus();
            input = this.DOM.content.find('input[name=\'orderconfirmmpwd\']')[0];
        },
        ok: function (here) {
            return yes && yes.call(this, inputaccount.value, input.value, here);
        },
        cancel: true
    });
};

function bargainReceiveConfirm(orderNo, rsakey) {
    art.artDialog.promptBargain(
        "砍价订单收货管理员确认",
        function (inputaccount, pwd) {
            var pwdRtn = encryptedString(rsakey, pwd);

            $.ajax({
                type: "POST",
                url: "/mobi/bargain/bargainReceiveConfirm",
                async: false,
                dataType: "text",
                data: "orderNo=" + orderNo + "&maccount=" + inputaccount + "&pwd=" + pwdRtn,
                success: function (msg) {
                    if (parseInt(msg) == 1000) {
                        location.reload();
                    }
                    else
                        art.dialog.tips(msg, null, 1.5);
                },
                error: function () {
                    alert('网络超时');
                }
            });
        },
        ""
        );
}