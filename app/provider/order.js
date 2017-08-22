/**
 * 订单相关接口
 */
import Base from './base';
import Payment from '../utils/pay';

class Order extends Base {

    constructor(...params) {
        super(...params);
    }

    /**
     * 微信支付
     * @param data
     * @param successCB
     * @param errorCB
     */
    weiXinPay(data, successCB, errorCB = null) {

        const defaultParams = [
            ['GetUserByOpenId']
        ];
        const finalParams = this.paramHandler(defaultParams, data);

        this.signRequest('WeiXinPay', finalParams, function (resultParams) {
            var payment = new Payment();
            successCB();
        }, errorCB);
    }

}

export default Order;