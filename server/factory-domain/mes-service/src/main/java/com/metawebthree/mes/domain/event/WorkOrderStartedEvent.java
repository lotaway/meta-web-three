package com.metawebthree.mes.domain.event;

public class WorkOrderStartedEvent extends MesEvent {
    private final Long workOrderId;

    public WorkOrderStartedEvent(Object source, Long workOrderId) {
        super(source, MesEventType.WORK_ORDER_STARTED);
        this.workOrderId = workOrderId;
    }

    public Long getWorkOrderId() {
        return workOrderId;
    }
}