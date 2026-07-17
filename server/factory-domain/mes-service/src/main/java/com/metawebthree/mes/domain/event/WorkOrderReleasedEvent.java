package com.metawebthree.mes.domain.event;

public class WorkOrderReleasedEvent extends MesEvent {
    private final Long workOrderId;

    public WorkOrderReleasedEvent(Long workOrderId) {
        super(MesEventType.WORK_ORDER_RELEASED);
        this.workOrderId = workOrderId;
    }

    public Long getWorkOrderId() {
        return workOrderId;
    }
}