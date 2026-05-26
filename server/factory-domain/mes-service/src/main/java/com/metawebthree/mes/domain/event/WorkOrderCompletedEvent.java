package com.metawebthree.mes.domain.event;

public class WorkOrderCompletedEvent extends MesEvent {
    private final Long workOrderId;

    public WorkOrderCompletedEvent(Object source, Long workOrderId) {
        super(source, MesEventType.WORK_ORDER_COMPLETED);
        this.workOrderId = workOrderId;
    }

    public Long getWorkOrderId() {
        return workOrderId;
    }
}