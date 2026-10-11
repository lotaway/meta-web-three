package com.metawebthree.payment.domain.exception;

public class DailyLimitExceededException extends RiskControlException {
    public DailyLimitExceededException(int dailyTotal, int limit) {
        super("Daily limit exceeded. Daily total: " + dailyTotal + ", Limit: " + limit);
    }
}