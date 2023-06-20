// <reference path="../config/init_config.ts"/>
import initConfig from "../config/init";
import host from "../config/host";
import {API_URL} from "../config/api";
import Decorator from "../utils/decorator"
import {BaseMapper, IApiMapper, IApiMapperStatic, MapperWrapper} from "./base"
import { ISystem } from "../core/iCore"

export namespace NSAdvertisement {

    interface BaseArgs {
        signal?: AbortSignal
    }

    export interface CategoryAdArgs extends BaseArgs {
        categoryIdentity: number | string     //  分类标识
        duoge?: number     //  是否多个 (0,不是;1是;默认不是)
        location: string        //  位置
        pageName: string            //  页面
    }

    export interface CategoryAdResponse extends BaseArgs {
        src: string
        link?: string
        url?: string
    }

    export interface GetPublicAdArgs extends BaseArgs {
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

    export type GetPublicAdsResponse = GetPublicAdResponse[]

    export interface GetAppHomeBannerArgs {
        num?: number
    }

    export interface GetAppStartAdParam extends BaseArgs {
        type?: string,  //  类型：[home:启动图，boot:引导图]
        num?: number    //  数量
    }

    interface AppImgItem extends BaseArgs {
        "@src": string
        ad: string
    }

    export interface GetAppStartAdResponse extends Array<AppImgItem> {
    }

    export class Adapter {

        @Decorator.AddHost(host.goodsService)
        static getCategoryAd(res: CategoryAdResponse) {
            if (res) {
                res.url = res.link
                delete res.link
            }
            return res
        }

        @Decorator.AddHost(host.goodsService)
        static getPublicAd(res: GetPublicAdsResponse) {
            return res
        }

        @Decorator.AddHost(host.goodsService, ["src"])
        static getAppHomeBanner(res: GetPublicAdsResponse) {
            return res
        }

        @Decorator.AddHost(host.goodsService, [["ad", '@src']])
        static getAppStartAd(res: GetAppStartAdResponse) {
            return res
        }
    }

    @Decorator.ImplementsWithStatic<IApiMapperStatic<IApiMapper<CategoryAdArgs, CategoryAdResponse>>>()
    export class CategoryAdMapper extends BaseMapper {
        @Decorator.DefaultArgs<[CategoryAdArgs]>({
            categoryIdentity: "",
            duoge: 0,
            location: "",
            pageName: ""
        })
        @Decorator.Sign(API_URL.getCategoryAd)
        @Decorator.UseAdapter(Adapter.getCategoryAd)
        async start(args: CategoryAdArgs): Promise<CategoryAdResponse> {
            super.init()
            return await this.rpc.request<CategoryAdResponse>(API_URL.getCategoryAd, args, this.options)
        }
    }

    @Decorator.ImplementsWithStatic<IApiMapperStatic<IApiMapper<GetPublicAdArgs, CategoryAdResponse>>>()
    export class PublicAdMapper extends BaseMapper {
        @Decorator.DefaultArgs<[GetPublicAdArgs]>({
            isMutiple: 0,
            location: "",
            name: ""
        })
        @Decorator.Sign(API_URL.getPublicAd)
        @Decorator.UseAdapter(Adapter.getPublicAd)
        async start<T = GetPublicAdsResponse | GetPublicAdResponse>(args: GetPublicAdArgs): Promise<T> {
            super.init()
            return await this.rpc.request<T>(API_URL.getPublicAd, args, this.options)
        }
    }

    @Decorator.ImplementsWithStatic<IApiMapperStatic<IApiMapper<GetAppHomeBannerArgs, GetPublicAdsResponse>>>()
    export class AppHomeBannerMapper extends BaseMapper {

        @Decorator.DefaultArgs({
            num: initConfig.SLIDER_IMG_NUM,
            page: 1,
            type: "top"
        })
        @Decorator.Sign(API_URL.getAppHomeBanner)
        @Decorator.UseAdapter(Adapter.getAppHomeBanner)
        async start(args: GetAppHomeBannerArgs): Promise<GetPublicAdsResponse> {
            super.init()
            return await this.rpc.request<GetPublicAdsResponse>(API_URL.getPublicAd, args, this.options)
        }
    }

    @Decorator.ImplementsWithStatic<IApiMapperStatic<IApiMapper<GetAppStartAdParam, GetAppStartAdResponse>>>()
    export class AppStartAdMapper extends BaseMapper {

        @Decorator.DefaultArgs({
            type: "home",
            num: 1
        })
        @Decorator.Sign(API_URL.getAppStartAd)
        @Decorator.UseAdapter(Adapter.getAppStartAd)
        async start(args: GetAppStartAdParam): Promise<GetAppStartAdResponse> {
            super.init()
            return await this.rpc.request<GetAppStartAdResponse>(API_URL.getAppStartAd, args, this.options)
        }
    }

    /**
     * 广告图接口
     */
    export class Service {

        // TODO 换成类装饰器和方法装饰器进行自动包裹？
        private readonly mapperWrapper: MapperWrapper

        constructor(systemImpl: ISystem) {
            this.mapperWrapper = new MapperWrapper(systemImpl)
        }

        get system(): ISystem {
            return this.mapperWrapper.system
        }

        async getCategoryAds<Args extends CategoryAdArgs, ResponseData>(args: Args, mapper: IApiMapper<Args, ResponseData>) {
            return await mapper.start(args)
        }

        async getCategoryAd(args: CategoryAdArgs) {
            return await this.mapperWrapper.start(CategoryAdMapper, args)
        }

        async getPublicAd<T = GetPublicAdsResponse | GetPublicAdResponse>(args: GetPublicAdArgs) {
            return await this.mapperWrapper.start<GetPublicAdArgs, T>(PublicAdMapper, args)
        }

        //  获取商标
        async getLogo() {
            return await this.getPublicAd({
                name: "home",
                location: "logo"
            })
        }

        async getAppHomeBanner(args: GetAppHomeBannerArgs = {}) {
            return await this.mapperWrapper.start(AppHomeBannerMapper, args)
        }

        async getAppStartAd(args: GetAppStartAdParam = {}) {
            return await this.mapperWrapper.start(AppStartAdMapper, args)
        }

    }
}
