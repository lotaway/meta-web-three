/**
 * App更新
 */
import Base from './base/base';

/**
 * APP检查更新
 * 参考http://ask.dcloud.net.cn/article/182 版本升级包
 * 参考http://ask.dcloud.net.cn/article/199 手动创建差量资源升级
 * @param {String} url -传入检查更新的地址
 */
class Update extends Base {

    constructor(...params){
        super(...params);
    }

    checkUpdate(url = this.host + "/app/version") {
        var app_version //	app版本号
            , waitingWindowObj = plus.nativeUI.showWaiting('正在获取数据')
            , self = this
            ;
        plus.runtime.getProperty(plus.runtime.appid, dev=> {
            app_version = dev.version;
            this.request({
                type: 'get',
                url: url,
                data: 'version=' + app_version,
                success: function (data) {
                    var url = ''; //	更新包下载地址
                    waitingWindowObj.close();
                    if (app_version < data.version) {
                        plus.nativeUI.toast('正在下载更新包');
                        url = data.updateUrl;
                        self.App.downLoad(url, {
                            successCB: function (download) {
                                self.App.installPackage(download.filename, {
                                    successCB: function () {
                                        plus.nativeUI.confirm('应用更新完成，是否立即重启？', function (event) {
                                            if (event.index === '0') plus.runtime.restart();
                                        }, '重启应用', ['立即重启', '暂不']);
                                    }
                                });
                            }
                        });
                    }
                    else plus.nativeUI.toast('已是最新版本');
                },
                error: function (e) {
                    self.errorHandler(e, function (e) {
                        plus.nativeUI.alert('获取更新失败：' + e);
                    });
                }
            });
        });
    }
}

export default Update;