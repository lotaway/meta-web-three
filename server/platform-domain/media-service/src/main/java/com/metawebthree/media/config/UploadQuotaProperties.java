package com.metawebthree.media.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.HashMap;
import java.util.Map;

@ConfigurationProperties(prefix = "upload")
public class UploadQuotaProperties {

    private Map<String, QuotaConfig> quotas = new HashMap<>();

    public Map<String, QuotaConfig> getQuotas() { return quotas; }
    public void setQuotas(Map<String, QuotaConfig> quotas) { this.quotas = quotas; }

    public static class QuotaConfig {
        private long maxFileSize = 10L * 1024 * 1024;
        private long totalQuota = 100L * 1024 * 1024;

        public long getMaxFileSize() { return maxFileSize; }
        public void setMaxFileSize(long maxFileSize) { this.maxFileSize = maxFileSize; }
        public long getTotalQuota() { return totalQuota; }
        public void setTotalQuota(long totalQuota) { this.totalQuota = totalQuota; }
    }
}
