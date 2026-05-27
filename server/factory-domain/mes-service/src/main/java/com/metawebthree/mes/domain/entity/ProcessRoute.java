package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

public class ProcessRoute {
    private Long id;
    private String routeCode;
    private String routeName;
    private String productCode;
    private Integer version;
    private RouteStatus status;
    private List<ProcessStep> steps;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum RouteStatus {
        DRAFT, ACTIVE, ARCHIVED
    }

    public static class ProcessStep {
        private Integer stepNo;
        private String processCode;
        private String processName;
        private String workstationId;
        private Integer standardTime;
        private String qualityCheckpoint;
        private Integer predecessorStepNo;
        private Integer successorStepNo;

        public Integer getStepNo() { return stepNo; }
        public void setStepNo(Integer stepNo) { this.stepNo = stepNo; }
        public String getProcessCode() { return processCode; }
        public void setProcessCode(String processCode) { this.processCode = processCode; }
        public String getProcessName() { return processName; }
        public void setProcessName(String processName) { this.processName = processName; }
        public String getWorkstationId() { return workstationId; }
        public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
        public Integer getStandardTime() { return standardTime; }
        public void setStandardTime(Integer standardTime) { this.standardTime = standardTime; }
        public String getQualityCheckpoint() { return qualityCheckpoint; }
        public void setQualityCheckpoint(String qualityCheckpoint) { this.qualityCheckpoint = qualityCheckpoint; }
        public Integer getPredecessorStepNo() { return predecessorStepNo; }
        public void setPredecessorStepNo(Integer predecessorStepNo) { this.predecessorStepNo = predecessorStepNo; }
        public Integer getSuccessorStepNo() { return successorStepNo; }
        public void setSuccessorStepNo(Integer successorStepNo) { this.successorStepNo = successorStepNo; }
    }

    public void create(String routeCode, String routeName, String productCode) {
        this.routeCode = routeCode;
        this.routeName = routeName;
        this.productCode = productCode;
        this.version = 1;
        this.status = RouteStatus.DRAFT;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.status = RouteStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void archive() {
        this.status = RouteStatus.ARCHIVED;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateVersion() {
        this.version++;
        this.updatedAt = LocalDateTime.now();
    }

    public ValidationResult validateSequence() {
        List<String> errors = new ArrayList<>();
        
        if (steps == null || steps.isEmpty()) {
            errors.add("工艺路线必须包含至少一个工序");
            return new ValidationResult(false, errors);
        }
        
        Set<Integer> stepNos = new HashSet<>();
        for (ProcessStep step : steps) {
            if (step.getStepNo() == null) {
                errors.add("工序序号不能为空");
                continue;
            }
            if (!stepNos.add(step.getStepNo())) {
                errors.add("工序序号 " + step.getStepNo() + " 重复");
            }
        }
        
        List<Integer> sortedStepNos = steps.stream()
            .map(ProcessStep::getStepNo)
            .filter(no -> no != null)
            .sorted()
            .collect(Collectors.toList());
        
        for (int i = 0; i < sortedStepNos.size() - 1; i++) {
            if (sortedStepNos.get(i + 1) - sortedStepNos.get(i) != 1) {
                errors.add("工序序号不连续: " + sortedStepNos.get(i) + " -> " + sortedStepNos.get(i + 1));
            }
        }
        
        Set<String> processCodes = new HashSet<>();
        for (ProcessStep step : steps) {
            if (step.getProcessCode() != null && !processCodes.add(step.getProcessCode())) {
                errors.add("工序编码 " + step.getProcessCode() + " 重复");
            }
        }
        
        for (ProcessStep step : steps) {
            if (step.getPredecessorStepNo() != null) {
                boolean predecessorExists = steps.stream()
                    .anyMatch(s -> step.getPredecessorStepNo().equals(s.getStepNo()));
                if (!predecessorExists) {
                    errors.add("工序 " + step.getStepNo() + " 的前驱工序 " + step.getPredecessorStepNo() + " 不存在");
                }
            }
            if (step.getSuccessorStepNo() != null) {
                boolean successorExists = steps.stream()
                    .anyMatch(s -> step.getSuccessorStepNo().equals(s.getStepNo()));
                if (!successorExists) {
                    errors.add("工序 " + step.getStepNo() + " 的后继工序 " + step.getSuccessorStepNo() + " 不存在");
                }
            }
        }
        
        return new ValidationResult(errors.isEmpty(), errors);
    }

    public Optional<ProcessStep> getNextStep(Integer currentStepNo) {
        if (steps == null || currentStepNo == null) {
            return Optional.empty();
        }
        
        // 按序号排序，找到当前工序的下一个
        List<ProcessStep> sortedSteps = steps.stream()
            .filter(s -> s.getStepNo() != null)
            .sorted(Comparator.comparing(ProcessStep::getStepNo))
            .collect(Collectors.toList());
        
        for (int i = 0; i < sortedSteps.size() - 1; i++) {
            if (sortedSteps.get(i).getStepNo().equals(currentStepNo)) {
                return Optional.of(sortedSteps.get(i + 1));
            }
        }
        
        return Optional.empty();
    }

    public Optional<ProcessStep> getStepByNo(Integer stepNo) {
        if (steps == null || stepNo == null) {
            return Optional.empty();
        }
        return steps.stream()
            .filter(s -> stepNo.equals(s.getStepNo()))
            .findFirst();
    }

    public Optional<ProcessStep> getFirstStep() {
        if (steps == null || steps.isEmpty()) {
            return Optional.empty();
        }
        return steps.stream()
            .filter(s -> s.getStepNo() != null)
            .min(Comparator.comparing(ProcessStep::getStepNo));
    }

    public Optional<ProcessStep> getLastStep() {
        if (steps == null || steps.isEmpty()) {
            return Optional.empty();
        }
        return steps.stream()
            .filter(s -> s.getStepNo() != null)
            .max(Comparator.comparing(ProcessStep::getStepNo));
    }

    public static class ValidationResult {
        private final boolean valid;
        private final List<String> errors;

        public ValidationResult(boolean valid, List<String> errors) {
            this.valid = valid;
            this.errors = errors;
        }

        public boolean isValid() { return valid; }
        public List<String> getErrors() { return errors; }
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRouteCode() { return routeCode; }
    public void setRouteCode(String routeCode) { this.routeCode = routeCode; }
    public String getRouteName() { return routeName; }
    public void setRouteName(String routeName) { this.routeName = routeName; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public Integer getVersion() { return version; }
    public void setVersion(Integer version) { this.version = version; }
    public RouteStatus getStatus() { return status; }
    public void setStatus(RouteStatus status) { this.status = status; }
    public List<ProcessStep> getSteps() { return steps; }
    public void setSteps(List<ProcessStep> steps) { this.steps = steps; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}