package com.metawebthree.mes.domain.event;

public class TaskStartedEvent extends MesEvent {
    private final Long taskId;

    public TaskStartedEvent(Object source, Long taskId) {
        super(source, MesEventType.TASK_STARTED);
        this.taskId = taskId;
    }

    public Long getTaskId() {
        return taskId;
    }
}