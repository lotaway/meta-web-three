package com.metawebthree.mes.infrastructure.event;

import com.metawebthree.common.event.DomainEventPublisher;
import com.metawebthree.mes.domain.event.*;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@Component
public class MesEventPublisher {

    private final DomainEventPublisher eventPublisher;

    public MesEventPublisher(DomainEventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishWorkOrderCreated(Long workOrderId, String workOrderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", workOrderId);
        data.put("workOrderNo", workOrderNo);
        data.put("eventType", MesEventType.WORK_ORDER_CREATED.name());
        eventPublisher.publish(MesEventType.WORK_ORDER_CREATED.name(), data);
        log.info("Published WorkOrderCreatedEvent: id={}, workOrderNo={}", workOrderId, workOrderNo);
    }

    public void publishWorkOrderReleased(Long workOrderId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", workOrderId);
        data.put("eventType", MesEventType.WORK_ORDER_RELEASED.name());
        eventPublisher.publish(MesEventType.WORK_ORDER_RELEASED.name(), data);
        log.info("Published WorkOrderReleasedEvent: id={}", workOrderId);
    }

    public void publishWorkOrderStarted(Long workOrderId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", workOrderId);
        data.put("eventType", MesEventType.WORK_ORDER_STARTED.name());
        eventPublisher.publish(MesEventType.WORK_ORDER_STARTED.name(), data);
        log.info("Published WorkOrderStartedEvent: id={}", workOrderId);
    }

    public void publishWorkOrderCompleted(Long workOrderId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", workOrderId);
        data.put("eventType", MesEventType.WORK_ORDER_COMPLETED.name());
        eventPublisher.publish(MesEventType.WORK_ORDER_COMPLETED.name(), data);
        log.info("Published WorkOrderCompletedEvent: id={}", workOrderId);
    }

    public void publishTaskCreated(Long taskId, String taskNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", taskId);
        data.put("taskNo", taskNo);
        data.put("eventType", MesEventType.TASK_CREATED.name());
        eventPublisher.publish(MesEventType.TASK_CREATED.name(), data);
        log.info("Published TaskCreatedEvent: id={}, taskNo={}", taskId, taskNo);
    }

    public void publishTaskStarted(Long taskId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", taskId);
        data.put("eventType", MesEventType.TASK_STARTED.name());
        eventPublisher.publish(MesEventType.TASK_STARTED.name(), data);
        log.info("Published TaskStartedEvent: id={}", taskId);
    }

    public void publishTaskCompleted(Long taskId, Integer qualified, Integer defective) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", taskId);
        data.put("qualified", qualified);
        data.put("defective", defective);
        data.put("eventType", MesEventType.TASK_COMPLETED.name());
        eventPublisher.publish(MesEventType.TASK_COMPLETED.name(), data);
        log.info("Published TaskCompletedEvent: id={}, qualified={}, defective={}", taskId, qualified, defective);
    }

    public void publishEquipmentBreakdown(Long equipmentId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", equipmentId);
        data.put("eventType", MesEventType.EQUIPMENT_BREAKDOWN.name());
        eventPublisher.publish(MesEventType.EQUIPMENT_BREAKDOWN.name(), data);
        log.info("Published EquipmentBreakdownEvent: id={}", equipmentId);
    }

    public void publishEquipmentRepaired(Long equipmentId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", equipmentId);
        data.put("eventType", MesEventType.EQUIPMENT_REPAIRED.name());
        eventPublisher.publish(MesEventType.EQUIPMENT_REPAIRED.name(), data);
        log.info("Published EquipmentRepairedEvent: id={}", equipmentId);
    }
}