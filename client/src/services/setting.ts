import initConfig from "../config/init"
import Base from './base'
import Decorator from "../utils/decorator";
import host from "../config/host";
import {API_URL, templateFolderPlaceHolder} from "../config/api";

namespace NSSetting {

    //  快捷入口实际数据
    export type FastEntryData = Array<{
        sort: number
        appId: string
        function_id: string
        icon: string
        is_enable: string
        linkType: string
        link_url: string
        openlink: string
        originId: string
        path: string
        title: string
    }>

    //  快捷入口响应数据
    export interface FastEntryResponse extends Object {
        navigation: {
            child_navigation: FastEntryData
        }
    }

    //  权限配置单项数据
    interface FunctionConfigItem extends Object {
        id: string
        is_enable: string
    }

    //  权限配置实际数据
    export interface FunctionConfigData extends Object {
        [id: string]: any
    }

    //  权限配置接口响应数据
    export interface FunctionConfigResponse extends Object {
        module: {
            info: Array<FunctionConfigItem>
        }
    }

    //  选项卡实际数据
    export interface TabBarData extends Object {
        list: Array<{
            title: string
            href?: string
            activeIcon?: string
            subNum?: number
            link: string
            icon: string
        }>
    }

    export interface TemplateModuleItemData {
        Name: string
        IsShow: boolean
        Title: string
        EnTitle: string
        Describe: string
        Remark: string
        Url: string
        Alt: string
        Src: string
        FieldType: string
        RadioItemsText?: string
        RadioItemsValue?: string
        RadioSelectedValue?: string
    }

    //  模板内设置
    export interface TemplateModuleResponse {
        ArrayOfModuleKeywordModel: Array<TemplateModuleItemData>
    }

    //  应用配置接口响应数据
    export interface GetAppConfigResponse {
        checkUpdate?: {
            url: string
        }
        cart: {
            addCartIcon?: string
        }
        index: {
            ad: {
                list: Array<{
                    sort: number
                    isShow: boolean
                    config: {
                        name: string
                        location: string
                    }
                    title: string
                    type: string
                    secTitle?: string
                }>
            }
            banner: {
                isShow: boolean
                sort: number
            }
            fastEntry: {
                isShow: boolean
                showType: string
                sort: number
                title: string
                presetList: Array<{
                    isShow: boolean
                    icon: string
                    title: string
                    url: string
                }>
            }
            filterGoods: {
                isShow: boolean
                sort: number
                showType: string
                pageSize?: number
                linkText: string
                list: Array<{
                    title: string
                    secTitle?: string
                    showType?: string
                    banner: {
                        query: {
                            name: string
                            location: string
                        }
                    }
                    apiFn: string
                    more: {
                        link: string
                        params: object
                        text?: string
                    }
                }>
            }
            notice: {
                sort: number
            }
            recommendGoods: {
                isShow: boolean
                showType: string
                sort: number
            }
            search: {
                isShow: boolean
                sort: number
                placeHolder: string
            }
            shop: {
                background: string
                isShow: boolean
                logo: {
                    config: {
                        location: string
                        name: string
                    }
                }
                sort: number
            }
        }
        status: number
        tab: TabBarData
        userCenter: {
            backgroundImage: string
            entry: {
                isShow: boolean
                title: string
            }
        }
        userMenu: {
            displayType: string
            list: {
                title: string
                id: string
                url: string
            }[]
        }
    }

    //  会员菜单实际数据
    export type GetUserMenuData = Array<{
        [key: string]: string
        Url: string
        Fid: string
    }>

    //  会员菜单响应数据
    export interface UserMenuResponse extends Object {
        ArrayOfMenuInfo: {
            MenuInfo: GetUserMenuData
        }
    }

    //  站点配置数据
    export interface GetSitePublicConfigData extends Object {
        customer_code: string
        IsShowVisitorPrice: string
        IntegralToPreDeposit: string
        microblog_code: string
        mobiMokor?: Array<string>
        MobiTemplate: string
        mobi_mokor_service_status: string
        mobi_mokor_code: string
        mokor_code: string
        webcountcode: string
        webcrod: string
        webname: string

        [key: string]: any
    }

    //  站点配置接口响应数据
    export interface SiteConfigResponse extends Object {
        Siteconfig: GetSitePublicConfigData
    }

    class Adapter {
        static getTabBar(tabBarData: GetAppConfigResponse['tab']) {
            if (!tabBarData || !tabBarData.list || tabBarData.list.length === 0) {
                tabBarData = {
                    list: [
                        {
                            link: "/",
                            icon: "icon-home",
                            title: "首页"
                        }
                        , {
                            link: "/user/collection",
                            icon: "icon-collect",
                            title: "收藏",
                        }
                        , {
                            link: "",
                            icon: "icon-category",
                            title: "分类"
                        }
                        , {
                            link: "/user/cart",
                            icon: "icon-cart",
                            title: "购物车"
                        }
                        , {
                            link: "/user/center",
                            icon: "icon-user",
                            title: "我的"
                        }
                    ]
                }
            }
            //  预处理数据
            try {
                tabBarData.list = (tabBarData as TabBarData).list.map(item => {
                    item.title = item.title ?? ""
                    item.href = item.link
                    item.activeIcon = item.activeIcon || (item.icon + "-fill")
                    item.subNum = 0
                    return item
                })
                return Promise.resolve(tabBarData)
            } catch (e) {
                return Promise.reject(e)
            }
        }

        @Decorator.AddHost(host.userService, [[["Url"]]])
        static getUserMenu(apiData: UserMenuResponse) {
            return apiData.ArrayOfMenuInfo.MenuInfo as GetUserMenuData
        }

        @Decorator.AddHost(host.userService, [[["link_url", "icon"]]])
        static getFastEntry(res: FastEntryResponse) {
            return res.navigation.child_navigation.filter(item => item.is_enable === "True").sort((prev, item) => prev.sort - item.sort).map(item => {
                switch (item.linkType) {
                    case "weChatMiniProgram":
                    case "thirdMiniProgram":
                        item.link_url = item.openlink
                        break
                    case "webLink":
                    default:
                        //	无需处理
                        break
                }
                return item
            }) as FastEntryData
        }

        static getTemplateConfig(responseData: TemplateModuleResponse) {
            let adapterData: {
                customerServiceInfo?: TemplateModuleItemData
                goodsListInfo?: TemplateModuleItemData
                salesGoodsInfo?: TemplateModuleItemData
                filterGoodsInfo?: TemplateModuleItemData
            } = {}
            responseData.ArrayOfModuleKeywordModel.forEach(item => {
                switch (item.Name) {
                    case "customer_mobile":
                        adapterData.customerServiceInfo = item
                        break
                    case "goods_list":
                        adapterData.goodsListInfo = item
                        break
                    case "is_show_goods_promote":
                        adapterData.filterGoodsInfo = item
                        break
                    case "is_show_goods_sales":
                        adapterData.salesGoodsInfo = item
                        break
                    default:
                        break
                }
            })
            return adapterData
        }

        @Decorator.AddHost(host.userService, [["backgroundImage", ["background", ['icon']]]])
        static getAppConfig(responseData: GetAppConfigResponse): GetAppConfigResponse {
            if (responseData.status === 1000) {
                responseData.userMenu.displayType = responseData.userMenu.displayType || ""
            }
            return responseData
        }

        static getFunctionConfig(res: FunctionConfigResponse): FunctionConfigData {
            let _data: FunctionConfigData = {};
            //  转换成以id为键，是否开启为值
            (res?.module?.info ?? []).forEach(item => {
                _data[Number(item.id)] = item.is_enable === "true"
            })
            _data[0] = true    //  无控制的内容默认开启
            return _data
        }

        @Decorator.AddHost(host.goodsService, [["noticeurl"]])
        static getSitePublicConfig(res: SiteConfigResponse): GetSitePublicConfigData {
            let _data = res.Siteconfig as Partial<GetSitePublicConfigData>
            const mobiMokorCodeMatches = (_data.mobi_mokor_code || "").match(/src="([^"]+)/g),
                needCustomerService = _data.mobi_mokor_service_status === "1" && mobiMokorCodeMatches
            if (needCustomerService)
                _data.mobiMokor = mobiMokorCodeMatches.map(item => item.replace(/^src="/, "").replace(/"$/, ""))
            else
                _data.mobiMokor = []
            delete _data.webcountcode
            delete _data.mokor_code
            delete _data.mobi_mokor_service_status
            delete _data.mobi_mokor_code
            delete _data.customer_code
            delete _data.microblog_code
            delete _data.webcountcode
            return _data as GetSitePublicConfigData
        }

    }

    /**
     * 配置文件接口
     */
    class Service extends Base {

        constructor() {
            super()
        }

        //  获取选项卡数据
        @Decorator.UseAdapter(Adapter.getTabBar)
        async getTabBar() {
            return await this.getAppConfig().then(appSetting => appSetting.status === 1000 ? Promise.resolve(appSetting.tab || {list: []}) : Promise.resolve({list: []}))
        }

        // 通过模板名称获取会员菜单
        @Decorator.UseAdapter(Adapter.getUserMenu)
        @Decorator.UseCache()
        async getUserMenu(args: { templateName: string }): Promise<ReturnType<typeof Adapter.getUserMenu>> {
            const finalApiUrl = API_URL.getUserMenu.replace(`{${templateFolderPlaceHolder}`, args.templateName)
            return await Service.request(finalApiUrl, {}, {
                method: "GET", dataType: "XML"
            })
        }

        /**
         * 获取首页菜单
         * @param args {object} 参数
         * @return {Promise} 是否成功
         */
        @Decorator.UseAdapter(Adapter.getFastEntry)
        async getHomeMenu(args: { templateName: string }): Promise<FastEntryData> {
            const finalApiUrl = API_URL.getHomeMenu.replace(`{${templateFolderPlaceHolder}}`, args.templateName)
            return await Service.request(finalApiUrl, {}, {method: "GET", dataType: "XML"})
        }

        /**
         * 获取模板配置
         * @param args {object} 参数
         * @return {Promise} 是否成功
         */
        @Decorator.UseAdapter(Adapter.getTemplateConfig)
        async getTemplateConfig(args: { templateName: string }) {
            const finalApiUrl = API_URL.getTemplateConfig.replace(`{${templateFolderPlaceHolder}`, args.templateName)
            return await Service.request(finalApiUrl, {}, {method: "GET", dataType: "XML"})
        }

        /**
         * 获取应用设置
         * @return {Promise} 是否成功
         */
        @Decorator.UseCache()
        @Decorator.UseAdapter(Adapter.getAppConfig)
        async getAppConfig(): Promise<GetAppConfigResponse> {
            return await Service.request(API_URL.getAppConfig, {}, {method: "GET", dataType: "XML"})
        }

        /**
         * 获取微信配置
         * @return {Promise} 是否成功
         */
        async getWeChatConfig() {
            return await Service.request(API_URL.getWeChatConfig, {}, {method: "GET", dataType: "XML"})
        }

        /**
         * 获取功能设置
         * @return {Promise} 是否成功
         */
        @Decorator.UseCache()
        @Decorator.UseAdapter(Adapter.getFunctionConfig)
        async getFunctionConfig(): Promise<{
            [functionId: number]: boolean
        }> {
            return await Service.request(API_URL.getFunctionConfig, {}, {method: "GET", dataType: "XML"})
        }

        /**
         * 获取站点配置
         * @return {Promise} 是否成功
         */
        @Decorator.UseCache()
        @Decorator.UseAdapter(Adapter.getSitePublicConfig)
        async getSitePublicConfig(): Promise<GetSitePublicConfigData> {
            return await Service.request(API_URL.getSitePublicConfig, {}, {method: "GET", dataType: "XML"})
        }

    }

}
