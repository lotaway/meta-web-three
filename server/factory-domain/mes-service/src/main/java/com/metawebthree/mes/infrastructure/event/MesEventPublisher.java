package com.metawebthree.mes.infrastructure.event;

import com.metawebthree.mes.domain.event.*;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Component;

@Component
public class MesEventPublisher {

    private final ApplicationEventPublisher eventPublisher;

    public MesEventPublisher(ApplicationEventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishWorkOrderCreated(Long workOrderId, String workOrderNo) {
        eventPublisher.publishEvent(new WorkOrderCreatedEvent(this, workOrderId, workOrderNo));
    }

    public void publishWorkOrderReleased(Long workOrderId) {
        eventPublisher.publishEvent(new WorkOrderReleasedEvent(this, workOrderId));
    }

    public void publishWorkOrderStarted(Long workOrderId) {
        eventPublisher.publishEvent(new WorkOrderStartedEvent(this, workOrderId));
    }

    public void publishWorkOrderCompleted(Long workOrderId) {
        eventPublisher.publishEvent(new WorkOrderCompletedEvent(this, workOrderId));
    }

    public void publishTaskCreated(Long taskId, String taskNo) {
        eventPublisher.publishEvent(new TaskCreatedEvent(this, taskId, taskNo));
    }

    public void publishTaskStarted(Long taskId) {
        eventPublisher.publishEvent(new TaskStartedEvent(this, taskId));
    }

    public void publishTaskCompleted(Long taskId, Integer qualified, Integer defective) {
        eventPublisher.publishEvent(new TaskCompletedEvent(this, taskId, qualified, defective));
    }

    public void publishEquipmentBreakdown(Long equipmentId) {
        eventPublisher.publishEvent(new EquipmentBreakdownEvent(this, equipmentId));
    }

    public void publishEquipmentRepaired(Long equipmentId) {
        eventPublisher.publishEvent(new EquipmentRepairedEvent(this, equipmentId));
    }
}