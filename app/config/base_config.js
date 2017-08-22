/**
 * APP内部配置
 */
const insetConfig = {
    "desKey": "Mnjo9SP8"  //  接口传递密文用DES秘钥，app和程序各一份
    , "desKeyUser": "DKMAB5DE"  //  用户cookies用DES秘钥，app和程序各一份
    , "userCookieName": 'id_member_shopbest' //    登录Cookie名称
    , "waitImgNum": 1       //  启动等待图数量
    , "bootImgNum": 3       //  引导图数量
    , "sliderImgNum": 5     //  首页轮播图数量
    , "goodsSalesNum": 4    //  猜你喜欢模块显示数量
    , "pageNum": 20         //  每页单项数量
    , "retryTime": 3    //  重试次数
    // ,"viewMainId": "HBuilder" //  主页窗口id，代码内部使用，真机调试时为HBuilder，打包后为应用名如：H5631BA51
    , "viewRemoteId": "view-remote" //  远程加载页面窗口id，代码内部使用
    , "viewRemoteChildId": "view-remote-child" //  远程加载页面窗口id，代码内部使用
    , "viewFunctionId": "view-function" //  功能类页面窗口id
    // ,"transitionTime": 300    //  过渡时间
    , "database": {   //  本地数据库设置
        name: "db_micronet_app"
        , version: 1
        , displayName: "微网商城本地数据库"
        , maxSize: 100000
        , opts: {}
    },
    //  推送消息设置
    pushMsgConfig: {
        defaultTitle: "推送消息"    //  默认标题
        ,defaultContent: "您有一条未读消息" //  默认内容
    }
};

export {insetConfig};