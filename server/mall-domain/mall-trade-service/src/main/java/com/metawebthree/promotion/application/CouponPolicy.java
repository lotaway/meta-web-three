package com.metawebthree.promotion.application;

public class CouponPolicy {
    private final int retryLimit;
    private final int maxGenerateCount;

    public CouponPolicy(int retryLimit, int maxGenerateCount) {
        this.retryLimit = retryLimit;
        this.maxGenerateCount = maxGenerateCount;
    }

    public int getRetryLimit() {
        return retryLimit;
    }

    public int getMaxGenerateCount() {
        return maxGenerateCount;
    }
}
