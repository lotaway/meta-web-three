import Base from './base';

/**
 * 广告图接口
 */
class Advertisement extends Base {

    constructor(...params) {
        super(...params);
    }

    /**
     * 获取分类广告
     * @param {Object} params 参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getCategoryAdverts(params = {}, successCB, errorCB = null) {
        const defaultParams = [
            ['categoryIdentity'],     //  分类标识
            ['duoge', 0],     //  是否多个 (0,不是;1是;默认不是)
            ['location'],        //  位置
            ['pageName']            //  页面
        ];
        const finalParams = this.paramHandler(defaultParams, params);
        this.signRequest('getCategoryAdverts', finalParams, data => {
            successCB(this.convertDataUrl(data));
        }, errorCB);
    }

    /**
     * 获取自定义广告
     * @param {Object} params 参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getAd(params = {}, successCB, errorCB = null) {
        const defaultParams = [
            ['isMutiple', 0],     //  是否多个 (0,不是;1是;默认不是)
            ['location'],        //  广告所处位置
            ['name']            //  大类所处页面名
        ];
        const finalParams = this.paramHandler(defaultParams, params);
        this.signRequest('GetAd', finalParams, data => {
            successCB(this.convertDataUrl(data));
        }, errorCB);
    }

    /**
     * 首页banner图(app专用)
     * @param {Number} num 图片数量
     * @param params 更多参数
     */
    getAdverInfoListTop(num = this.App.getConfig("sliderImgNum"), ...params) {
        this.signRequest('AdverInfoList', `num=${num}&page=1&type=Top`, ...params);
    }

    /**
     * 获取启动图/引导图
     * @param {Object} query 参数
     * @param params 更多参数
     */
    getAppImg(query = {}, ...params) {
        const defaultParams = {
            type: "home"    //  [home:启动图，boot:引导图]
            , num: 1    //  数量
        };
        const finalParams = this.paramHandler(defaultParams, query);
        this.signRequest('AppImg', finalParams, ...params);
    }
}

export default Advertisement;