package com.metawebthree.common.trace;

/**
 * Bottleneck analysis result
 */
public class BottleneckAnalysis {
    private String traceId;
    private boolean bottleneckDetected;
    private String bottleneckService;
    private String bottleneckMethod;
    private double bottleneckPercentage;
    private long totalDurationMs;
    private int spanCount;
    private TraceRecord slowestSpan;
    
    public String getTraceId() {
        return traceId;
    }
    
    public void setTraceId(String traceId) {
        this.traceId = traceId;
    }
    
    public boolean isBottleneckDetected() {
        return bottleneckDetected;
    }
    
    public void setBottleneckDetected(boolean bottleneckDetected) {
        this.bottleneckDetected = bottleneckDetected;
    }
    
    public String getBottleneckService() {
        return bottleneckService;
    }
    
    public void setBottleneckService(String bottleneckService) {
        this.bottleneckService = bottleneckService;
    }
    
    public String getBottleneckMethod() {
        return bottleneckMethod;
    }
    
    public void setBottleneckMethod(String bottleneckMethod) {
        this.bottleneckMethod = bottleneckMethod;
    }
    
    public double getBottleneckPercentage() {
        return bottleneckPercentage;
    }
    
    public void setBottleneckPercentage(double bottleneckPercentage) {
        this.bottleneckPercentage = bottleneckPercentage;
    }
    
    public long getTotalDurationMs() {
        return totalDurationMs;
    }
    
    public void setTotalDurationMs(long totalDurationMs) {
        this.totalDurationMs = totalDurationMs;
    }
    
    public int getSpanCount() {
        return spanCount;
    }
    
    public void setSpanCount(int spanCount) {
        this.spanCount = spanCount;
    }
    
    public TraceRecord getSlowestSpan() {
        return slowestSpan;
    }
    
    public void setSlowestSpan(TraceRecord slowestSpan) {
        this.slowestSpan = slowestSpan;
    }
}
