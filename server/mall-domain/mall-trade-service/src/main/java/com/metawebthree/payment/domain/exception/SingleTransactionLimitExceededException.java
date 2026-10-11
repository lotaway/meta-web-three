package com.metawebthree.payment.domain.exception;

public class SingleTransactionLimitExceededException extends RiskControlException {
    public SingleTransactionLimitExceededException(int limit, int actual) {
        super("Single transaction limit exceeded. Limit: " + limit + ", Actual: " + actual);
    }
}