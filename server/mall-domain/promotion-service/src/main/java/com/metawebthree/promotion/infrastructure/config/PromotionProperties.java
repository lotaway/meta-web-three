package com.metawebthree.promotion.infrastructure.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "promotion.coupon")
public class PromotionProperties {
    private int retryLimit;
    private int maxGenerateCount;
    private int codeLength;
    private String codeAlphabet;

    public int getRetryLimit() { return retryLimit; }
    public void setRetryLimit(int retryLimit) { this.retryLimit = retryLimit; }
    public int getMaxGenerateCount() { return maxGenerateCount; }
    public void setMaxGenerateCount(int maxGenerateCount) { this.maxGenerateCount = maxGenerateCount; }
    public int getCodeLength() { return codeLength; }
    public void setCodeLength(int codeLength) { this.codeLength = codeLength; }
    public String getCodeAlphabet() { return codeAlphabet; }
    public void setCodeAlphabet(String codeAlphabet) { this.codeAlphabet = codeAlphabet; }
}
