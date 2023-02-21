// <reference path="../config/init_config.ts"/>
import initConfig from "../config/init";
import host from "../config/host";
import {API_URL} from "../config/api";
import Decorator from "../utils/decorator"
import BaseService from "./base"

export namespace NSAdvertisement {

    export interface CategoryAdArgs extends Object {
        categoryIdentity: number | string     //  分类标识
        duoge?: number     //  是否多个 (0,不是;1是;默认不是)
        location: string        //  位置
        pageName: string            //  页面
    }

    export interface CategoryAdResponse extends Object {
        src: string
        link?: string
        url?: string
    }

    export interface GetPublicAdArgs extends Object {
        isMutiple?: number  //  是否多个 (0,不是;1是;默认不是)
        location: string    //  广告所处位置
        name: string    //  大类所处页面名
    }

    export type GetPublicAdResponse = {
        alt: string
        height: string
        src: string
        url: string
        width: string
    }

    export type GetPublicAdsResponse = Array<GetPublicAdResponse>

    export interface GetAppHomeBannerArgs {
        num?: number
    }

    export interface GetAppStartAdParam extends Object {
        type?: string,  //  类型：[home:启动图，boot:引导图]
        num?: number    //  数量
    }

    interface AppImgItem extends Object {
        "@src": string
        ad: string
    }

    export interface GetAppStartAdResponse extends Array<AppImgItem> {
    }

    export class Adapter {

        @Decorator.setUrlWithHost(host.goodsService)
        static getCategoryAd(res: CategoryAdResponse) {
            if (res) {
                res.url = res.link
                delete res.link
            }
            return res
        }

        @Decorator.setUrlWithHost(host.goodsService)
        static getPublicAd(res: GetPublicAdsResponse) {
            return res
        }

        @Decorator.setUrlWithHost(host.goodsService, ["src"])
        static getAppHomeBanner(res: GetPublicAdsResponse) {
            return res
        }

        @Decorator.setUrlWithHost(host.goodsService, [["ad", '@src']])
        static getAppStartAd(res: GetAppStartAdResponse) {
            return res
        }
    }

    /**
     * 广告图接口
     */
    export class Service extends BaseService {

        constructor() {
            super()
        }

        /**
         * 获取分类广告图
         * @param {Object} args 参数
         * @return {Promise} 是否成功
         */
        @Decorator.setDefaultArgs<[CategoryAdArgs]>({
            categoryIdentity: "",
            duoge: 0,
            location: "",
            pageName: ""
        })
        @Decorator.sign(API_URL.getCategoryAd)
        @Decorator.useAdapter(Adapter.getCategoryAd)
        async getCategoryAd(args: CategoryAdArgs) {
            return await Service.request<CategoryAdResponse>(API_URL.getCategoryAd, args)
        }

        /**
         * 获取自定义广告
         * @param {Object} args 参数
         * @param {string=} apiUrl 接口路径
         * @return {Promise} 是否成功
         */
        @Decorator.setDefaultArgs<[GetPublicAdArgs]>({
            isMutiple: 0,
            location: "",
            name: ""
        })
        @Decorator.sign(API_URL.getPublicAd)
        @Decorator.useAdapter(Adapter.getPublicAd)
        async getPublicAd<T = GetPublicAdsResponse | GetPublicAdResponse>(args: GetPublicAdArgs, apiUrl = API_URL.getPublicAd) {
            return await Service.request<T>(apiUrl, args)
        }

        //  获取商标
        async getLogo() {
            return await this.getPublicAd({
                name: "home",
                location: "logo"
            });
        }

        /**
         * 获取应用轮播图
         * @param {object} args 参数
         * @return {Promise} 是否成功
         */
        @Decorator.setDefaultArgs({
            num: initConfig.SLIDER_IMG_NUM,
            page: 1,
            type: "top"
        })
        @Decorator.sign(API_URL.getAppHomeBanner)
        @Decorator.useAdapter(Adapter.getAppHomeBanner)
        async getAppHomeBanner(args: GetAppHomeBannerArgs = {}) {
            return await Service.request<GetPublicAdsResponse>(API_URL.getAppHomeBanner, args)
        }

        /**
         * 获取启动图/引导图
         * @param {Object} args 参数
         * @param params 更多参数
         * @return {Promise} 是否成功
         */
        @Decorator.setDefaultArgs({
            type: "home",
            num: 1
        })
        @Decorator.sign(API_URL.getAppStartAd)
        @Decorator.useAdapter(Adapter.getAppStartAd)
        async getAppStartAd(args: GetAppStartAdParam = {}, ...params: any) {
            return await Service.request<GetAppStartAdResponse>(API_URL.getAppStartAd, args)
        }

    }
}
