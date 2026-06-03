package com.metawebthree.common.trace;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a service node in the service dependency topology
 */
public class ServiceNode {
    private String serviceName;
    private int callCount;
    private long totalDurationMs;
    private long maxDurationMs;
    private long avgDurationMs;
    private int errorCount;
    private List<String> downstreamServices = new ArrayList<>();

    public ServiceNode() {}

    public ServiceNode(String serviceName) {
        this.serviceName = serviceName;
    }

    public String getServiceName() {
        return serviceName;
    }

    public void setServiceName(String serviceName) {
        this.serviceName = serviceName;
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

    public long getMaxDurationMs() {
        return maxDurationMs;
    }

    public void setMaxDurationMs(long maxDurationMs) {
        this.maxDurationMs = maxDurationMs;
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

    public List<String> getDownstreamServices() {
        return downstreamServices;
    }

    public void setDownstreamServices(List<String> downstreamServices) {
        this.downstreamServices = downstreamServices;
    }

    public void addDownstreamService(String service) {
        if (!downstreamServices.contains(service)) {
            downstreamServices.add(service);
        }
    }

    public double getErrorRate() {
        if (callCount == 0) return 0.0;
        return (double) errorCount / callCount * 100.0;
    }
}
