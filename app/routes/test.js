var http = require("http")
    , querystring = require("querystring")
    , crypto = require('crypto')
    , Promise = require('Promise')
    ;

function start(req, res) {
    res.render('testIndex', {
        page: {
            // title: title
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

// 个推测试
function getui(req, res) {
    res.render('getui');
}

//  个推接口
var auth_token = "";    //  权限令牌
async function api(req, res) {

    var backstageData = {   //  后台填写信息
            appid: "YnIrfHY8bT9cAopJYTanHA",
            appkey: "Ls0pmQWqTP9bFA22pGD5Y2",   //  注册应用时生成的appkey
            mastersecret: "U1zHiTzCBB7xfVZeWKpbq9",
            clientId: "573afa01b9e3d68fe0c7432e1dc02221"
        },
        reqData = { //  用于发送请求的参数
            message: {
                appkey: backstageData.appkey,
                // is_offline: false,   //  是否离线推送
                // offline_expire_time: 2 * 60 * 60 * 1000, // 消息离线存储有效期，单位：ms
                // push_network_type: 0,   //  选择推送消息使用网络类型，0：不限制，1：wifi
                msgtype: "notification" //  消息应用类型，可选项：notification、link、notypopload、transmission*/
            },
            notification: {
                /*text	String	是	消息展示正文
                 title	String	是	消息展示标题
                 logo	String	否	消息展示logo
                 logourl	String	否	消息展示logourl
                 transmission_type	boolean	否	收到消息是否立即启动应用，true为立即启动，false则广播等待启动，默认是否
                 transmission_content	String	否	透传内容
                 is_ring	boolean	否	是否声音提示，默认值true
                 is_vibrate	boolean	否	是否振动提示，默认值true
                 is_clearable	boolean	否	是否可消除，默认值true
                 duration_begin	String	否	设定展示开始时间，格式为yyyy-MM-dd HH:mm:ss
                 duration_end	String	否	设定展示结束时间，格式为yyyy-MM-dd HH:mm:ss
                 notify_style	integer	否	通知栏消息布局样式(0 系统样式 1 个推样式) 默认为0*/
                title: req.body.title,
                text: req.body.content
            },
            cid: backstageData.clientId,    //  与alias二选一
            // alias: backstageData.alias,    //  与cid二选一
            requestid: ""   //  请求唯一标识
        }
        , pushSingleData = "" //  获取的内容
        ,
        reqObj;
    //  进行鉴权
    if (auth_token === "") {

        let hasher = crypto.createHash("sha256")  //  加密算法
            , authSignReqData = {   //  鉴权请求参数
            timestamp: Date.now(),
            appkey: backstageData.appkey
        }
            , result;

        hasher.update(authSignReqData.appkey + authSignReqData.timestamp + backstageData.mastersecret);
        authSignReqData.sign = hasher.digest("hex");
        authSignReqData = querystring.stringify(authSignReqData);

        auth_token = await new Promise(function (resolve, reject) {
            var authSignResData = ""   //  返回的数据
                , reqObj = http.request({
                host: 'https://restapi.getui.com',
                // port: 443,
                path: `/v1/${backstageData.appid}/auth_sign`,
                method: 'POST',
                headers: {
                    "Content-Type": 'application/x-www-form-urlencoded',
                    "Content-Length": authSignReqData.length
                }
            }, function (resHttp) {
                if (resHttp.statusCode == 200) {
                    resHttp
                        .on("data", function (dataPiece) {
                            authSignResData += dataPiece;
                        })
                        .on("end", function () {
                            authSignResData = JSON.parse(authSignResData);
                            switch (authSignResData.result) {
                                case "ok":
                                    resolve(authSignResData.auth_token);
                                    break;
                                default:
                                    reject(authSignResData);
                                    break;
                            }
                        });
                }
                else {
                    reject(authSignResData);
                }
            });
            reqObj.write(authSignReqData);
            reqObj.end();
        });
        // auth_token = authSignRequest(backstageData);
    }
    reqData.requestid = auth_token;
    reqData = querystring.stringify(reqData);

    reqObj = http.request({
        host: 'https://restapi.getui.com',
        // port: 443,
        path: `/v1/${backstageData.appid}/push_single`,
        method: 'POST',
        headers: {
            "Content-Type": 'application/x-www-form-urlencoded',
            "Content-Length": reqData.length
        }
    }, function (resHttp) {
        if (resHttp.statusCode == 200) {
            resHttp
                .on("data", function (dataPiece) {
                    pushSingleData += dataPiece;
                })
                .on("end", function () {
                    pushSingleData = JSON.parse(pushSingleData);
                    switch (pushSingleData.result) {
                        case "ok":
                            res.send(200, pushSingleData);
                            break;
                        default:
                            res.send(200, "推送失败");
                            break;
                    }
                });
        }
        else {
            res.send(500, "推送请求失败");
        }
    });
    reqObj.write(reqData);
    reqObj.end();

}

/**
 * 鉴权请求
 * @param backstageData {Object} 后台填写的数据
 * @return {Promise.<void>}
 */
async function authSignRequest(backstageData) {
}

exports.start = start;
exports.getui = getui;
exports.api = api;