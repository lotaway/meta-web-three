/**
 * 一元购商品相关接口
 */
import Base from './base/base';

class CrowdFunding extends Base {

    constructor(...params) {
        super(...params);
    }

    /**
     * 获取热销云购商品
     * @param {Object} params 请求参数
     * @param {Function} callback 成功回调
     * @param {Function} errorCB 失败回调
     */
    getHotGoods(params = {}, callback, errorCB = null) {
        params.sort = 1;
        params.filter = 2;
        this.getGoods(params, callback, errorCB);
    }

    /**
     * 获取已揭晓的云购商品
     * @param {Object} params 请求参数
     * @param {Function} callback 成功回调
     * @param {Function} errorCB 失败回调
     */
    /*static getHistoryGoods(params = {}, callback, errorCB = null) {
        params.sort = 2;
        CrowdFunding.getGoods(params, callback, errorCB);
    }*/

    /**
     * 获取最新揭晓的云购商品
     * @param {Object} params 请求参数
     * @param {Function} callback 成功回调
     * @param {Function} errorCB 失败回调
     */
    getNewGoods(params = {}, callback, errorCB = null) {
        params.dt = "winTime";
        params.sort = 2;
        this.getGoods(params, callback, errorCB);
    }

    /**
     * 获取云购商品列表
     * @param {Object} params 请求参数
     * @param {Function} callback 成功回调
     * @param {Function} errorCB 失败回调
     */
    getGoods(params = {}, callback, errorCB = null) {

        const defaultParams = {
            sort: 1,      //  排序 [1 进行中,2 已揭晓]
            cid: "",        //  筛选 分类ID
            bid: "",        //  筛选 品牌ID
            p: 1,           //  筛选 页码
            kw: "",         //  筛选 关键词
            price: "",      //  筛选 价格区间
            dt: "",         //  排序？ [restrate ？,price 价格高到底,priceAsc 价格低到高,id 最新,winTime 按揭晓时间]
            filter: 1,      //  筛选 [2 热销]
            act: 'ajaxDropLoad' //  获取数据（否则是输出整个页面？）
        };
        const finalParams = this.paramHandler(defaultParams, params);
        this.request({
            url: this.getApiUrl("getCrowdFundingGoods"),
            data: finalParams,
            success: callback,
            error: errorCB
        });
    }

    toCart(params = {}, callback, errorCB = null) {

        const defaultParams = {
            count: 1,           //  数量
            goodsEntitys: -1, //  商品实体id
            act: "add"          //  类型 添加商品
        };
        const finalParams = this.paramHandler(defaultParams, params);
        this.request({
            url: this.getApiUrl("handleCrowdFundingShopping"),
            data: finalParams,
            success: function (msg) {
                switch (msg) {
                    case 1000:
                        // plus.nativeUI.toast("已添加到购物车");
                        break;
                    case 1002:
                        plus.nativeUI.toast("添加失败，商品本期次数不足！");
                        break;
                    case 10031:
                        plus.nativeUI.toast("添加失败，您已达到限购人次！");
                        break;
                }
                callback(msg);
            },
            error: errorCB
        });
    }

}

export default CrowdFunding;