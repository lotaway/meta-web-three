package com.metawebthree.mes.domain.event;

public class WorkOrderCreatedEvent extends MesEvent {
    private final Long workOrderId;
    private final String workOrderNo;

    public WorkOrderCreatedEvent(Long workOrderId, String workOrderNo) {
        super(MesEventType.WORK_ORDER_CREATED);
        this.workOrderId = workOrderId;
        this.workOrderNo = workOrderNo;
    }

    public Long getWorkOrderId() {
        return workOrderId;
    }

    public String getWorkOrderNo() {
        return workOrderNo;
    }
}