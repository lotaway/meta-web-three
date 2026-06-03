package com.metawebthree.common.aitest;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents an AI-generated test scenario
 * Contains test steps, expected results, and performance expectations
 */
public class TestScenario {
    private String scenarioId;
    private String name;
    private String description;
    private String category; // PERFORMANCE, RELIABILITY, FUNCTIONAL, STRESS
    private String priority; // HIGH, MEDIUM, LOW
    private List<TestStep> steps = new ArrayList<>();
    private PerformanceExpectation performanceExpectation;
    private List<String> prerequisites = new ArrayList<>();
    private String targetService;

    public String getScenarioId() {
        return scenarioId;
    }

    public void setScenarioId(String scenarioId) {
        this.scenarioId = scenarioId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getPriority() {
        return priority;
    }

    public void setPriority(String priority) {
        this.priority = priority;
    }

    public List<TestStep> getSteps() {
        return steps;
    }

    public void setSteps(List<TestStep> steps) {
        this.steps = steps;
    }

    public PerformanceExpectation getPerformanceExpectation() {
        return performanceExpectation;
    }

    public void setPerformanceExpectation(PerformanceExpectation performanceExpectation) {
        this.performanceExpectation = performanceExpectation;
    }

    public List<String> getPrerequisites() {
        return prerequisites;
    }

    public void setPrerequisites(List<String> prerequisites) {
        this.prerequisites = prerequisites;
    }

    public String getTargetService() {
        return targetService;
    }

    public void setTargetService(String targetService) {
        this.targetService = targetService;
    }
}
