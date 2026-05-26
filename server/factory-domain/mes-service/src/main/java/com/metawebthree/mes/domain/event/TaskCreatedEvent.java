package com.metawebthree.mes.domain.event;

public class TaskCreatedEvent extends MesEvent {
    private final Long taskId;
    private final String taskNo;

    public TaskCreatedEvent(Object source, Long taskId, String taskNo) {
        super(source, MesEventType.TASK_CREATED);
        this.taskId = taskId;
        this.taskNo = taskNo;
    }

    public Long getTaskId() {
        return taskId;
    }

    public String getTaskNo() {
        return taskNo;
    }
}