package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import com.metawebthree.mes.domain.QcConstants;

public class NonConformanceDisposition {
    private Long id;
    private String dispositionCode;
    private String dispositionName;
    private DispositionType type;
    private List<DispositionStep> steps;
    private Boolean isEnabled;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum DispositionType {
        SCRAP, REWORK, REPAIR, RETURN, USE_AS_IS, SPECIAL_ACCEPTANCE
    }

    public static class DispositionStep {
        private Integer stepOrder;
        private String stepName;
        private StepAction action;
        private String assigneeRole;
        private Boolean requiresApproval;
        private Integer timeoutHours;

        public enum StepAction {
            IDENTIFY, ISOLATE, EVALUATE, DECIDE, EXECUTE, VERIFY, CLOSE
        }

        public Integer getStepOrder() { return stepOrder; }
        public void setStepOrder(Integer stepOrder) { this.stepOrder = stepOrder; }
        public String getStepName() { return stepName; }
        public void setStepName(String stepName) { this.stepName = stepName; }
        public StepAction getAction() { return action; }
        public void setAction(StepAction action) { this.action = action; }
        public String getAssigneeRole() { return assigneeRole; }
        public void setAssigneeRole(String assigneeRole) { this.assigneeRole = assigneeRole; }
        public Boolean getRequiresApproval() { return requiresApproval; }
        public void setRequiresApproval(Boolean requiresApproval) { this.requiresApproval = requiresApproval; }
        public Integer getTimeoutHours() { return timeoutHours; }
        public void setTimeoutHours(Integer timeoutHours) { this.timeoutHours = timeoutHours; }
    }

    public void create(String dispositionCode, String dispositionName, DispositionType type) {
        this.dispositionCode = dispositionCode;
        this.dispositionName = dispositionName;
        this.type = type;
        this.steps = new ArrayList<>();
        this.isEnabled = Boolean.TRUE;
        this.sortOrder = QcConstants.DEFAULT_SORT_ORDER;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void addStep(DispositionStep.StepAction action, String stepName, String assigneeRole,
                        Boolean requiresApproval, Integer timeoutHours) {
        if (this.steps == null) {
            this.steps = new ArrayList<>();
        }
        DispositionStep step = new DispositionStep();
        step.setStepOrder(this.steps.size() + 1);
        step.setStepName(stepName);
        step.setAction(action);
        step.setAssigneeRole(assigneeRole);
        step.setRequiresApproval(requiresApproval);
        step.setTimeoutHours(timeoutHours);
        this.steps.add(step);
        this.updatedAt = LocalDateTime.now();
    }

    public void addDefaultScrapFlow() {
        this.steps = new ArrayList<>();
        addStep(DispositionStep.StepAction.IDENTIFY, "标识不合格品", "QC", Boolean.FALSE, QcConstants.SCRAP_TIMEOUT_IDENTIFY);
        addStep(DispositionStep.StepAction.ISOLATE, "隔离不合格品", "WAREHOUSE", Boolean.FALSE, QcConstants.SCRAP_TIMEOUT_ISOLATE);
        addStep(DispositionStep.StepAction.EVALUATE, "评审不合格品", "ENGINEER", Boolean.TRUE, QcConstants.SCRAP_TIMEOUT_EVALUATE);
        addStep(DispositionStep.StepAction.DECIDE, "确定处置方式", "QM", Boolean.TRUE, QcConstants.SCRAP_TIMEOUT_DECIDE);
        addStep(DispositionStep.StepAction.EXECUTE, "执行处置", "WAREHOUSE", Boolean.FALSE, QcConstants.SCRAP_TIMEOUT_EXECUTE);
        addStep(DispositionStep.StepAction.VERIFY, "验证处置结果", "QC", Boolean.TRUE, QcConstants.SCRAP_TIMEOUT_VERIFY);
        addStep(DispositionStep.StepAction.CLOSE, "关闭不合格品处理单", "QC", Boolean.FALSE, QcConstants.SCRAP_TIMEOUT_CLOSE);
    }

    public void addDefaultReworkFlow() {
        this.steps = new ArrayList<>();
        addStep(DispositionStep.StepAction.IDENTIFY, "标识不合格品", "QC", Boolean.FALSE, QcConstants.REWORK_TIMEOUT_IDENTIFY);
        addStep(DispositionStep.StepAction.ISOLATE, "隔离不合格品", "WAREHOUSE", Boolean.FALSE, QcConstants.REWORK_TIMEOUT_ISOLATE);
        addStep(DispositionStep.StepAction.EVALUATE, "评估返工可行性", "ENGINEER", Boolean.TRUE, QcConstants.REWORK_TIMEOUT_EVALUATE);
        addStep(DispositionStep.StepAction.DECIDE, "批准返工", "QM", Boolean.TRUE, QcConstants.REWORK_TIMEOUT_DECIDE);
        addStep(DispositionStep.StepAction.EXECUTE, "执行返工", "PRODUCTION", Boolean.FALSE, QcConstants.REWORK_TIMEOUT_EXECUTE);
        addStep(DispositionStep.StepAction.VERIFY, "返工后检验", "QC", Boolean.TRUE, QcConstants.REWORK_TIMEOUT_VERIFY);
        addStep(DispositionStep.StepAction.CLOSE, "关闭处理单", "QC", Boolean.FALSE, QcConstants.REWORK_TIMEOUT_CLOSE);
    }

    public Optional<DispositionStep> findStep(DispositionStep.StepAction action) {
        if (steps == null) {
            return Optional.empty();
        }
        return steps.stream()
            .filter(s -> action.equals(s.getAction()))
            .findFirst();
    }

    public void removeStep(Integer stepOrder) {
        if (steps != null) {
            steps.removeIf(s -> stepOrder.equals(s.getStepOrder()));
            reorderSteps();
            this.updatedAt = LocalDateTime.now();
        }
    }

    public void reorderSteps() {
        if (steps == null) {
            return;
        }
        for (int i = 0; i < steps.size(); i++) {
            steps.get(i).setStepOrder(i + 1);
        }
    }

    public void disable() {
        this.isEnabled = Boolean.FALSE;
        this.updatedAt = LocalDateTime.now();
    }

    public void enable() {
        this.isEnabled = Boolean.TRUE;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateSortOrder(Integer order) {
        this.sortOrder = order;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getDispositionCode() { return dispositionCode; }
    public void setDispositionCode(String dispositionCode) { this.dispositionCode = dispositionCode; }
    public String getDispositionName() { return dispositionName; }
    public void setDispositionName(String dispositionName) { this.dispositionName = dispositionName; }
    public DispositionType getType() { return type; }
    public void setType(DispositionType type) { this.type = type; }
    public List<DispositionStep> getSteps() { return steps; }
    public void setSteps(List<DispositionStep> steps) { this.steps = steps; }
    public Boolean getIsEnabled() { return isEnabled; }
    public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}