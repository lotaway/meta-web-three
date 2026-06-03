package com.metawebthree.common.aitest;

/**
 * Performance expectation for a test scenario
 */
public class PerformanceExpectation {
    private long maxResponseTimeMs;
    private long avgResponseTimeMs;
    private double maxErrorRatePercent;
    private int minThroughputRps;
    private long maxP99LatencyMs;

    public long getMaxResponseTimeMs() {
        return maxResponseTimeMs;
    }

    public void setMaxResponseTimeMs(long maxResponseTimeMs) {
        this.maxResponseTimeMs = maxResponseTimeMs;
    }

    public long getAvgResponseTimeMs() {
        return avgResponseTimeMs;
    }

    public void setAvgResponseTimeMs(long avgResponseTimeMs) {
        this.avgResponseTimeMs = avgResponseTimeMs;
    }

    public double getMaxErrorRatePercent() {
        return maxErrorRatePercent;
    }

    public void setMaxErrorRatePercent(double maxErrorRatePercent) {
        this.maxErrorRatePercent = maxErrorRatePercent;
    }

    public int getMinThroughputRps() {
        return minThroughputRps;
    }

    public void setMinThroughputRps(int minThroughputRps) {
        this.minThroughputRps = minThroughputRps;
    }

    public long getMaxP99LatencyMs() {
        return maxP99LatencyMs;
    }

    public void setMaxP99LatencyMs(long maxP99LatencyMs) {
        this.maxP99LatencyMs = maxP99LatencyMs;
    }
}
