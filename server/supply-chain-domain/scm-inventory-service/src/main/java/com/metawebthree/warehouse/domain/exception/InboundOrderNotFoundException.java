package com.metawebthree.warehouse.domain.exception;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;

/**
 * 入库单不存在异常
 */
public class InboundOrderNotFoundException extends BusinessException {

    public InboundOrderNotFoundException(String orderNo, String message) {
        super(ResponseStatus.INBOUND_ORDER_NOT_FOUND, message != null ? message : "入库单号[" + orderNo + "]不存在");
    }

    public InboundOrderNotFoundException(String orderNo) {
        this(orderNo, null);
    }
}