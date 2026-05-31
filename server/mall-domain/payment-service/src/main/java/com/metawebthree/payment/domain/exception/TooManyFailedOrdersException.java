package com.metawebthree.payment.domain.exception;

public class TooManyFailedOrdersException extends RiskControlException {
    public TooManyFailedOrdersException(int count) {
        super("Too many failed orders. Failed count: " + count);
    }
}