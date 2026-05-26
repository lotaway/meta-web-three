package com.metawebthree.mes.domain.event;

public class TaskCompletedEvent extends MesEvent {
    private final Long taskId;
    private final Integer qualifiedQuantity;
    private final Integer defectiveQuantity;

    public TaskCompletedEvent(Object source, Long taskId, Integer qualifiedQuantity, Integer defectiveQuantity) {
        super(source, MesEventType.TASK_COMPLETED);
        this.taskId = taskId;
        this.qualifiedQuantity = qualifiedQuantity;
        this.defectiveQuantity = defectiveQuantity;
    }

    public Long getTaskId() {
        return taskId;
    }

    public Integer getQualifiedQuantity() {
        return qualifiedQuantity;
    }

    public Integer getDefectiveQuantity() {
        return defectiveQuantity;
    }
}