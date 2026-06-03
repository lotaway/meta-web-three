package com.metawebthree.common.trace;

import java.util.ArrayList;
import java.util.List;

/**
 * Root cause analysis result for performance bottlenecks
 * Identifies the likely root cause from a chain of slow spans
 */
public class RootCauseAnalysis {
    private String traceId;
    private boolean rootCauseFound;
    private String rootCauseService;
    private String rootCauseMethod;
    private String rootCauseType; // SLOW, ERROR, TIMEOUT, HIGH_ERROR_RATE
    private String rootCauseDetail;
    private long rootCauseDurationMs;
    private double impactPercentage; // how much this root cause contributes to total latency
    private List<String> affectedDownstreamServices = new ArrayList<>();
    private List<String> diagnosisSteps = new ArrayList<>();
    private List<TraceRecord> suspectSpans = new ArrayList<>();

    public String getTraceId() {
        return traceId;
    }

    public void setTraceId(String traceId) {
        this.traceId = traceId;
    }

    public boolean isRootCauseFound() {
        return rootCauseFound;
    }

    public void setRootCauseFound(boolean rootCauseFound) {
        this.rootCauseFound = rootCauseFound;
    }

    public String getRootCauseService() {
        return rootCauseService;
    }

    public void setRootCauseService(String rootCauseService) {
        this.rootCauseService = rootCauseService;
    }

    public String getRootCauseMethod() {
        return rootCauseMethod;
    }

    public void setRootCauseMethod(String rootCauseMethod) {
        this.rootCauseMethod = rootCauseMethod;
    }

    public String getRootCauseType() {
        return rootCauseType;
    }

    public void setRootCauseType(String rootCauseType) {
        this.rootCauseType = rootCauseType;
    }

    public String getRootCauseDetail() {
        return rootCauseDetail;
    }

    public void setRootCauseDetail(String rootCauseDetail) {
        this.rootCauseDetail = rootCauseDetail;
    }

    public long getRootCauseDurationMs() {
        return rootCauseDurationMs;
    }

    public void setRootCauseDurationMs(long rootCauseDurationMs) {
        this.rootCauseDurationMs = rootCauseDurationMs;
    }

    public double getImpactPercentage() {
        return impactPercentage;
    }

    public void setImpactPercentage(double impactPercentage) {
        this.impactPercentage = impactPercentage;
    }

    public List<String> getAffectedDownstreamServices() {
        return affectedDownstreamServices;
    }

    public void setAffectedDownstreamServices(List<String> affectedDownstreamServices) {
        this.affectedDownstreamServices = affectedDownstreamServices;
    }

    public List<String> getDiagnosisSteps() {
        return diagnosisSteps;
    }

    public void setDiagnosisSteps(List<String> diagnosisSteps) {
        this.diagnosisSteps = diagnosisSteps;
    }

    public List<TraceRecord> getSuspectSpans() {
        return suspectSpans;
    }

    public void setSuspectSpans(List<TraceRecord> suspectSpans) {
        this.suspectSpans = suspectSpans;
    }
}
