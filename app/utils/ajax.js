import crypto from './crypto';
import {App} from '../utils/app';

/**
 * 基础接口封装
 */
class Ajax {

    constructor(...params) {
        this.host = App.getConfig("host");
    }

    /**
     * 在数据集中转化网址
     * @param data {Object|Array} 数据集
     * @param attrArray {Array|String} 要检查的属性名
     * @return {Object|Array} 返回的数据集
     */
    convertDataUrl(data, attrArray = ['src', 'url']) {
        if (!attrArray) return null;
        if (typeof attrArray === "string") attrArray = [attrArray];
        if (data.length) {
            data.map(item => this.convertDataUrl(item, attrArray));
        }
        else if (typeof data === 'object') {
            attrArray.forEach(attr => {
                data[attr] = this.convertUrl(data[attr]);
            });
        }
        return data;
    }

    /**
     * 转化网址（缺少域名时添加）
     * @param url (String) 需要转化的地址
     * @return url (String) 转化后的地址
     */
    convertUrl(url) {
        if (/^\//.test(url)) {
            url = this.host + url;
        }
        return url;
    }

    /**
     * 暂时代替Object.assign，待采用垫片polyfill测试
     * @param defaultObj {Object} 默认参数对象
     * @param realObj {Object} 新参数对象
     * @return {Object} 最终参数
     */
    assign(defaultObj = {}, realObj = {}) {
        for (let p in realObj) {
            defaultObj[p] = realObj[p];
        }
        return defaultObj;
    }

    /**
     * 实际参数和默认参数处理成序列
     * @param {Array|Object} defaultParams 默认参数数组（因为对排序有要求，没有采用对象形式）
     * @param {Object} realParams 实际传入的数据对象
     * @returns {string} result 返回序列字符串
     */
    paramHandler(defaultParams = [], realParams = {}) {
        var result = '';
        //  数组类型，意味着参数有顺序要求
        if (defaultParams.hasOwnProperty(length) || defaultParams.length === 0) {
            defaultParams.forEach(function (item) {
                //  key: item[0], defaultValue: item[0], realValue: realParams[key]
                if (realParams[item[0]] !== undefined) {
                    result += "&" + item[0] + '=' + realParams[item[0]];
                }
                else result += "&" + ((item.length >= 2) ? item.join("=") : (item[0] + '='));
            });
        }
        //  对象类型，意味着参数没有顺序要求
        else if (typeof defaultParams === 'object') {
            for (let item in defaultParams) {
                if (defaultParams.hasOwnProperty(item)) {
                    if (realParams[item] !== undefined) {
                        result += "&" + item + "=" + realParams[item];
                    }
                    else result += "&" + item + '=' + defaultParams[item];
                }
            }
        }
        return result.replace(/^&/, "");
    }

    /**
     * 异步请求
     * @param {Object} params 异步参数对象
     */
    request(params = {}) {
        var xhr = new plus.net.XMLHttpRequest(), //	异步对象
            responseText = {}; //	响应内容
        if (!params.url) return null;
        params.type = params.type || 'POST';
        params.data = params.data || '';
        params.dataType = params.dataType || 'JSON';
        if (params.timeout) xhr.timeout = params.timeout;
        xhr.onreadystatechange = function () {
            if (typeof params.change === 'function') params.change(xhr.readyState);
            if (xhr.readyState === 4) {
                // console.info(params.url + "  " + xhr.readyState + "  " + xhr.status);
                if (xhr.status === 200) {
                    switch (params.dataType) {
                        case 'JSON':
                            try {
                                responseText = JSON.parse(xhr.responseText);
                            }
                            catch (e) {
                                responseText = xhr.responseText;
                            }
                            break;
                        case "XML":
                            // todo 在ajax类里添加公用的xml转json格式，方便统一处理
                            responseText = xhr.responseText;
                            break;
                        default:
                            responseText = xhr.responseText;
                            break;
                    }
                    if (typeof params.success === 'function') {
                        params.success(responseText);
                    }
                }
                else {
                    App.errorHandler({xhr: xhr, _customerErrorMsg: "接口返回状态错误"}, function () {
                        if (typeof params.error === 'function') {
                            params.error(xhr);
                        }
                    });
                }
                if (typeof params.complete === 'function') {
                    params.complete(responseText, xhr);
                }
            }
        };
        xhr.open(params.type, params.url + (params.type === "GET" ? ("?" + params.data) : ""));
        xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        xhr.send(params.data);
        return xhr;
    }

}

export {Ajax};