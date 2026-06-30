package com.metawebthree.common.trace;


/**
 * Represents an edge (call relationship) between two services in the topology
 */
public class ServiceEdge {
    private String sourceService;
    private String targetService;
    private int callCount;
    private long totalDurationMs;
    private long avgDurationMs;
    private int errorCount;

    public ServiceEdge() {}

    public ServiceEdge(String sourceService, String targetService) {
        this.sourceService = sourceService;
        this.targetService = targetService;
    }

    public String getSourceService() {
        return sourceService;
    }

    public void setSourceService(String sourceService) {
        this.sourceService = sourceService;
    }

    public String getTargetService() {
        return targetService;
    }

    public void setTargetService(String targetService) {
        this.targetService = targetService;
    }

    public int getCallCount() {
        return callCount;
    }

    public void setCallCount(int callCount) {
        this.callCount = callCount;
    }

    public long getTotalDurationMs() {
        return totalDurationMs;
    }

    public void setTotalDurationMs(long totalDurationMs) {
        this.totalDurationMs = totalDurationMs;
    }

    public long getAvgDurationMs() {
        return avgDurationMs;
    }

    public void setAvgDurationMs(long avgDurationMs) {
        this.avgDurationMs = avgDurationMs;
    }

    public int getErrorCount() {
        return errorCount;
    }

    public void setErrorCount(int errorCount) {
        this.errorCount = errorCount;
    }

    public double getErrorRate() {
        if (callCount == 0) return 0.0;
        return (double) errorCount / callCount * 100.0;
    }
}
