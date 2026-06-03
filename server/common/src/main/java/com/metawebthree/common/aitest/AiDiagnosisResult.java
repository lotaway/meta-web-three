package com.metawebthree.common.aitest;

import java.util.ArrayList;
import java.util.List;

/**
 * AI root cause diagnosis result
 * Analyzes trace data, metrics, and logs to identify root causes
 */
public class AiDiagnosisResult {
    private String diagnosisId;
    private String traceId;
    private boolean diagnosisSuccessful;
    private String rootCauseSummary;
    private String rootCauseType; // SLOW_QUERY, RESOURCE_EXHAUSTION, CASCADE_FAILURE, CONFIGURATION_ERROR, NETWORK_ISSUE
    private String affectedService;
    private String affectedMethod;
    private List<String> evidenceChain = new ArrayList<>();
    private List<String> remediationSteps = new ArrayList<>();
    private List<String> similarHistoricalIncidents = new ArrayList<>();
    private double confidenceScore;

    public String getDiagnosisId() {
        return diagnosisId;
    }

    public void setDiagnosisId(String diagnosisId) {
        this.diagnosisId = diagnosisId;
    }

    public String getTraceId() {
        return traceId;
    }

    public void setTraceId(String traceId) {
        this.traceId = traceId;
    }

    public boolean isDiagnosisSuccessful() {
        return diagnosisSuccessful;
    }

    public void setDiagnosisSuccessful(boolean diagnosisSuccessful) {
        this.diagnosisSuccessful = diagnosisSuccessful;
    }

    public String getRootCauseSummary() {
        return rootCauseSummary;
    }

    public void setRootCauseSummary(String rootCauseSummary) {
        this.rootCauseSummary = rootCauseSummary;
    }

    public String getRootCauseType() {
        return rootCauseType;
    }

    public void setRootCauseType(String rootCauseType) {
        this.rootCauseType = rootCauseType;
    }

    public String getAffectedService() {
        return affectedService;
    }

    public void setAffectedService(String affectedService) {
        this.affectedService = affectedService;
    }

    public String getAffectedMethod() {
        return affectedMethod;
    }

    public void setAffectedMethod(String affectedMethod) {
        this.affectedMethod = affectedMethod;
    }

    public List<String> getEvidenceChain() {
        return evidenceChain;
    }

    public void setEvidenceChain(List<String> evidenceChain) {
        this.evidenceChain = evidenceChain;
    }

    public List<String> getRemediationSteps() {
        return remediationSteps;
    }

    public void setRemediationSteps(List<String> remediationSteps) {
        this.remediationSteps = remediationSteps;
    }

    public List<String> getSimilarHistoricalIncidents() {
        return similarHistoricalIncidents;
    }

    public void setSimilarHistoricalIncidents(List<String> similarHistoricalIncidents) {
        this.similarHistoricalIncidents = similarHistoricalIncidents;
    }

    public double getConfidenceScore() {
        return confidenceScore;
    }

    public void setConfidenceScore(double confidenceScore) {
        this.confidenceScore = confidenceScore;
    }
}
