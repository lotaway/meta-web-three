import Ajax from '../utils/ajax';
import App from '../utils/app';

class Base extends Ajax {

    constructor(...params) {
        super(...params);
        this.App = App;
    }

    /**
     * 通用接口封装加密（业务）
     * @param {String} apiName 接口名称
     * @param {String} data 数据
     * @param {Function} successCB 回调
     * @param {Function} errorCB 回调
     * @param {String} noSignData 无须附加到加密验证字符串里的参数
     * @param {String} type 请求类型
     */
    signRequest(apiName, data = "", successCB, errorCB = null, noSignData = "", type = "POST") {
        const queryData = (data + (noSignData ? "&" + noSignData : "") + "&sign=" + crypto.Md5.init(data.replace(/[=&]/g, "") + apiName)).replace(/^&/, "")
            ,self = this
            ;
        var config = {
            type: 'POST',
            url: this.getApiUrl(apiName),
            success: function (data) {
                switch (data) {
                    case 'TypeError':
                        self.errorHandler("类型出错");
                        break;
                    case 'error':
                        self.errorHandler("接口报错");
                        break;
                    case 'SignError':
                        self.errorHandler("验证不通过");
                        break;
                    case "ParaError":
                        self.errorHandler("参数有误");
                        break;
                    case 'CategoryError':
                        self.errorHandler("分类有误");
                        break;
                    case 'DataError':
                        self.errorHandler("查询数据有误");
                        break;
                    case 'PageError':
                        self.errorHandler("页码大于数据量页码");
                        break;
                    default:
                        successCB(data);
                        break;
                }
            },
            error: errorCB
        };
        switch (type) {
            case "POST":
                config.data = queryData;
                break;
            case "GET":
                config.url += "?" + queryData;
                break;
            default:
                self.errorHandler("无法处理的请求类型");
                break;
        }
        this.request(config);
    }

    /**
     * 查询接口路径
     * @param apiName {String} 接口名称
     * @return url {string} 接口路径
     */
    getApiUrl(apiName) {
        var url    //  接口路径
            , host = this.host //   域名
        ;

        switch (apiName) {
            //  根据分类标识获取子分类
            case "getCategoryByParentId":
                url = `${host}/api/category/categoryByParentId`;
                break;
            //  获取云购商品列表
            case "getCrowdFundingGoods":
                url = `${host}/mobi/cn/CrowdfundGoods_list.html`;
                break;
            //  云购购物车操作
            case "handleCrowdFundingShopping":
                url = `${host}/ashx/CrowdfundGoods_shoppingHandle.ashx`;
                break;
            //  获取商品列表
            case "getGoods":
                url = `${host}/ashx/cn/goods.ashx`;
                break;
            //    微信配置
            case "weChatConfig":
                url = "/Config/public/weixin_config.xml";
                break;
            //    站点设置
            case "siteConfig":
                url = "/Config/public/site.xml";
                break;
            //    功能设置
            case "functionConfig":
                url = "/Config/public/function.xml";
                break;
            //    默认的规范接口形式
            default:
                url = `${host}/api/app/${apiName}.api`;
                break;
        }

        return url;
    }

    /**
     * 错误处理
     * @param params
     */
    errorHandler(...params) {
        this.App.errorHandler(...params);
    }

}

export default Base;