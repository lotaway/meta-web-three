package com.metawebthree.common.aitest;

import java.util.ArrayList;
import java.util.List;

/**
 * AI-generated optimization suggestion based on historical performance data
 */
public class OptimizationSuggestion {
    private String suggestionId;
    private String category; // PERFORMANCE, RELIABILITY, SCALABILITY, COST
    private String severity; // CRITICAL, HIGH, MEDIUM, LOW
    private String title;
    private String description;
    private String currentBehavior;
    private String suggestedChange;
    private String expectedImprovement;
    private String targetService;
    private String targetMethod;
    private List<String> evidence = new ArrayList<>();
    private double confidenceScore; // 0.0 - 1.0

    public String getSuggestionId() {
        return suggestionId;
    }

    public void setSuggestionId(String suggestionId) {
        this.suggestionId = suggestionId;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getSeverity() {
        return severity;
    }

    public void setSeverity(String severity) {
        this.severity = severity;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getCurrentBehavior() {
        return currentBehavior;
    }

    public void setCurrentBehavior(String currentBehavior) {
        this.currentBehavior = currentBehavior;
    }

    public String getSuggestedChange() {
        return suggestedChange;
    }

    public void setSuggestedChange(String suggestedChange) {
        this.suggestedChange = suggestedChange;
    }

    public String getExpectedImprovement() {
        return expectedImprovement;
    }

    public void setExpectedImprovement(String expectedImprovement) {
        this.expectedImprovement = expectedImprovement;
    }

    public String getTargetService() {
        return targetService;
    }

    public void setTargetService(String targetService) {
        this.targetService = targetService;
    }

    public String getTargetMethod() {
        return targetMethod;
    }

    public void setTargetMethod(String targetMethod) {
        this.targetMethod = targetMethod;
    }

    public List<String> getEvidence() {
        return evidence;
    }

    public void setEvidence(List<String> evidence) {
        this.evidence = evidence;
    }

    public double getConfidenceScore() {
        return confidenceScore;
    }

    public void setConfidenceScore(double confidenceScore) {
        this.confidenceScore = confidenceScore;
    }
}
