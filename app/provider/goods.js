import Base from './base';
import 'babel-polyfill';

/**
 * 普通商品相关接口
 */
class Goods extends Base {

    constructor(...params) {
        super(...params);
    }

    /**
     * 获取猜你喜欢商品（即按销量排序）
     * @param {Number} top 数量
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getGoodsSalesCharts(top = this.App.getConfig("goodsSalesNum"), successCB, errorCB = null) {
        this.signRequest("GetGoodsSalesCharts", `top=${top}`, successCB, errorCB);
    }

    /**
     * 商品列表接口
     * @param {Object} params 请求参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 失败回调
     */
    getGoods(params = {}, successCB, errorCB = null) {

        const defaultParams = {
            sort: "1",      //  排序 [1,2,3,4,5,6,7,8,9]
            cid: "",        //  筛选 分类ID
            bid: "",        //  筛选 品牌ID
            p: 1,           //  筛选 页码
            kw: "",         //  筛选 关键词
            price: "",      //  筛选 价格区间
            minDate: "",    //  筛选 最早创建时间？
            maxDate: "",    //  筛选 最迟创建时间？
            gn: "",
            // dt: "",         //  布局 显示方式[image,list]
            purchase: "",   //  筛选 是否限购[0，1]
            opay: '',
            isba: '',       //  筛选 是否可砍价[0,1]
            cashBack: ''    //  筛选 是否返现[0,1]
        };
        const finalParams = this.paramHandler(defaultParams, params);
        this.request({
            url: this.getApiUrl("getGoods"),
            data: finalParams,
            dataType: 'json',
            success: successCB,
            error: errorCB
        });
    }

    /**
     * 普通商品列表
     * @param {Object} params 请求参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getGoodsList(params = {}, successCB, errorCB = null) {
        //  默认选项
        const defaultData = [
            ["num", this.App.getConfig('pageNum')],
            ["order", "Sort"],  //  排序类型，类型有：Sort按排序字段排序、Time按修改时间排序、Price按价格排序、Count按销量排序、Click按点击数排序
            ["orderValue", 0],  //  排序值，0为升序asc，1为降序desc
            ["page", 1],        //  获取第几页数据,0为全部输出
            ["type", "All"],    //  查询数据类型（Promote--促销产品，All--所有产品、Brand品牌、Category分类、Name商品名称 ）可以用”,"分开
            ["typeValue"]       //  Promote的1为推荐，Brand的为id，Category的为id，All可以为空，Name为商品的名称goodsName
        ];
        const finalParams = this.paramHandler(defaultData, params);

        this.signRequest('GoodsInfoListForAllCategory', finalParams, data => {
            successCB(this.convertDataUrl(data, ["thumbImg"]));
        }, errorCB);
    }

    /**
     * 普通商品列表 促销
     * @param {Object} params 请求参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getGoodsListPromote(params = {}, successCB, errorCB = null) {

        const defaultData = {
            "type": "Promote",
            "typeValue": 1
        };
        const finalParams = this.assign(defaultData, params);

        this.getGoodsList(finalParams, successCB, errorCB);
    }

    /**
     * 普通商品列表 最新
     * @param {Object} params 请求参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getGoodsListNew(params = {}, successCB, errorCB = null) {
        const defaultData = {
            "type": "New",
            "typeValue": 1
        };
        const finalParams = this.assign(defaultData, params);
        this.getGoodsList(finalParams, successCB, errorCB);
    }

    /**
     * 普通商品列表 精品推荐
     * @param {Object} params 请求参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getGoodsListBest(params = {}, successCB, errorCB = null) {
        const defaultData = {
            "type": "Best",
            "typeValue": 1
        };
        const finalParams = this.assign(defaultData, params);
        this.getGoodsList(finalParams, successCB, errorCB);
    }

    /**
     * 普通商品列表 热销
     * @param {Object} params 请求参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getGoodsListTopSelling(params = {}, successCB, errorCB = null) {
        const defaultData = {
            "type": "TopSelling",
            "typeValue": 1
        };
        const finalParams = this.assign(defaultData, params);
        this.getGoodsList(finalParams, successCB, errorCB);
    }

    /**
     * 普通商品列表 时间排序
     * @param {Object} params 请求参数
     * @param {Function} callback 成功回调
     * @param {Function} errorCB 错误回调
     */
    getGoodsListTime(params = {}, callback, errorCB = null) {
        const defaultData = {
            "order": "Time"
        };
        const finalParams = this.assign(defaultData, params);
        this.getGoodsList(finalParams, callback, errorCB);
    }

    /**
     * 普通商品列表 销量排序
     * @param {Object} params 请求参数
     * @param {Function} callback 成功回调
     * @param {Function} errorCB 错误回调
     */
    getGoodsListCount(params = {}, callback, errorCB = null) {
        const defaultData = {
            "order": "Count"
        };
        const finalParams = this.assign(defaultData, params);
        this.getGoodsList(finalParams, callback, errorCB);
    }

    /**
     * 获取所有分类和从属的品牌 或者 获取所有品牌和属下的分类
     * @param {Object} params 请求参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getCategoryBrand(params = {}, successCB, errorCB = null) {
        const defaultData = {
            type: "Category" // Category--分类主体数据，Brand--品牌主体数据
        };
        const finalParams = this.paramHandler(defaultData, params);
        this.signRequest('CategoryBrand', finalParams, successCB, errorCB);
    }

    /**
     * 根据父分类id 获取子分类列表
     * @param {Object} params 请求参数
     * @param {Function} successCB 成功回调
     * @param {Function} errorCB 错误回调
     */
    getCategoryByParentId(params = {}, successCB, errorCB = null) {
        this.request({
            url: this.getApiUrl("getCategoryByParentId"),
            data: `pId=${params.pId}`,
            dataType: 'JSON',
            success: successCB,
            error: errorCB
        });
    }

}

export default Goods;