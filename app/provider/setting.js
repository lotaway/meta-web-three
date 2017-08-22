import Base from './base';

/**
 * 配置文件接口
 */

class Setting extends Base {

    constructor(...params) {
        super(...params);
    }

    /**
     * 获取微信配置
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getWeChatConfig(successCB, errorCB = null) {
        this.request({
            dataType: "XML",
            url: this.getApiUrl("weChatConfig"),
            success: function (xml) {
                successCB(xml);
            },
            error: errorCB
        });
    }

}

export default Setting;