import Base from './base';

/**
 * 云购用的倒计时，使用接口获取时间
 * @param {String|Object} target 倒计时显示的对象容器
 * @param {String} endTime 结束时间
 * @param {Function} callback 完成倒计时后的回调
 */
function countDown(target, endTime = "", callback = () => {
}) {

    var base = new Base();

    if (typeof target === 'string') {
        target = document.querySelectorAll(target);
    }
    const host = base.host;
    [].forEach.call(target, function (item) {
        var o = {
            hm: item.querySelector(".hm"),
            sec: item.querySelector(".sec"),
            mini: item.querySelector(".mini"),
            hour: item.querySelector(".hour"),
            day: item.querySelector(".day"),
            month: item.querySelector(".month"),
            year: item.querySelector(".year")
        };

        base.request({
            url: host + "/ashx/CrowdfundGoods_shoppingHandle.ashx?acttype=getTimeOut&period=" + item.dataset["period"],
            success: function (msg) {
                var endDate = new Date()
                    ;
                endDate.setSeconds(endDate.getSeconds() + parseInt(msg));

                var f = {
                    timer: null,
                    haomiao: function (n) {
                        if (n < 10)return "0" + n.toString();
                        return n.toString();
                    },
                    zero: function (n) {
                        var _n = parseInt(n, 10);   //解析字符串,返回整数
                        if (_n > 0) {
                            if (_n <= 9) {
                                _n = "0" + _n;
                            }
                            return String(_n);
                        } else {
                            return "00";
                        }
                    },
                    dv: function () {
                        //d = d || Date.UTC(2050, 0, 1); //如果未定义时间，则我们设定倒计时日期是2050年1月1日
                        var _d = item.dataset["end"] || endTime;
                        var now = new Date();//,
                        //endDate = new Date(_d);
                        //现在将来秒差值
                        //alert(future.getTimezoneOffset());
                        var dur = (endDate - now.getTime()) / 1000, mss = endDate - now.getTime(), pms = {
                            hm: "000",
                            sec: "00",
                            mini: "00",
                            hour: "00",
                            day: "00",
                            month: "00",
                            year: "0"
                        };
                        if (mss > 0) {
                            pms.hm = f.haomiao(Math.floor(dur * 100 % 100));
                            pms.sec = f.zero(dur % 60);
                            pms.mini = Math.floor((dur / 60)) > 0 ? f.zero(Math.floor((dur / 60)) % 60) : "00";
                            pms.hour = Math.floor((dur / 3600)) > 0 ? f.zero(Math.floor((dur / 3600)) % 24) : "00";
                            pms.day = Math.floor((dur / 86400)) > 0 ? f.zero(Math.floor((dur / 86400)) % 30) : "00";
                            //月份，以实际平均每月秒数计算
                            pms.month = Math.floor((dur / 2629744)) > 0 ? f.zero(Math.floor((dur / 2629744)) % 12) : "00";
                            //年份，按按回归年365天5时48分46秒算
                            pms.year = Math.floor((dur / 31556926)) > 0 ? Math.floor((dur / 31556926)) : "0";
                        } else {
                            clearTimeout(f.timer);
                            callback(item);
                        }
                        return pms;
                    },
                    ui: function () {
                        var pmsObj = f.dv();
                        if (o.hm) {
                            o.hm.innerHTML = pmsObj.hm;
                        }
                        if (o.sec) {
                            o.sec.innerHTML = pmsObj.sec;
                        }
                        if (o.mini) {
                            o.mini.innerHTML = pmsObj.mini;
                        }
                        if (o.hour) {
                            o.hour.innerHTML = pmsObj.hour;
                        }
                        if (o.day) {
                            o.day.innerHTML = pmsObj.day;
                        }
                        if (o.month) {
                            o.month.innerHTML = pmsObj.month;
                        }
                        if (o.year) {
                            o.year.innerHTML = pmsObj.year;
                        }
                        f.timer = setTimeout(f.ui, 1);
                    }
                };
                /*f end */
                f.ui();
            }
        });
    });
}
export default countDown;