package com.metawebthree.mes.infrastructure.event;

import com.metawebthree.common.event.DomainEventPublisher;
import com.metawebthree.mes.domain.event.*;
import com.metawebthree.mes.domain.model.OutboxEventDO;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionalEventListener;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@Component
public class MesEventPublisher {

    private final DomainEventPublisher eventPublisher;

    public MesEventPublisher(DomainEventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishWorkOrderCreated(Long workOrderId, String workOrderNo, String eventId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", workOrderId);
        data.put("workOrderNo", workOrderNo);
        data.put("eventType", MesEventType.WORK_ORDER_CREATED.name());
        data.put("eventId", eventId); // Include eventId for consumer idempotency
        eventPublisher.publish(MesEventType.WORK_ORDER_CREATED.name(), data);
        log.info("Published WorkOrderCreatedEvent: id={}, workOrderNo={}, eventId={}", workOrderId, workOrderNo, eventId);
    }

    public void publishWorkOrderReleased(Long workOrderId, String eventId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", workOrderId);
        data.put("eventType", MesEventType.WORK_ORDER_RELEASED.name());
        data.put("eventId", eventId);
        eventPublisher.publish(MesEventType.WORK_ORDER_RELEASED.name(), data);
        log.info("Published WorkOrderReleasedEvent: id={}, eventId={}", workOrderId, eventId);
    }

    public void publishWorkOrderStarted(Long workOrderId, String eventId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", workOrderId);
        data.put("eventType", MesEventType.WORK_ORDER_STARTED.name());
        data.put("eventId", eventId);
        eventPublisher.publish(MesEventType.WORK_ORDER_STARTED.name(), data);
        log.info("Published WorkOrderStartedEvent: id={}, eventId={}", workOrderId, eventId);
    }

    public void publishWorkOrderCompleted(Long workOrderId, String eventId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", workOrderId);
        data.put("eventType", MesEventType.WORK_ORDER_COMPLETED.name());
        data.put("eventId", eventId);
        eventPublisher.publish(MesEventType.WORK_ORDER_COMPLETED.name(), data);
        log.info("Published WorkOrderCompletedEvent: id={}, eventId={}", workOrderId, eventId);
    }

    public void publishTaskCreated(Long taskId, String taskNo, String eventId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", taskId);
        data.put("taskNo", taskNo);
        data.put("eventType", MesEventType.TASK_CREATED.name());
        data.put("eventId", eventId);
        eventPublisher.publish(MesEventType.TASK_CREATED.name(), data);
        log.info("Published TaskCreatedEvent: id={}, taskNo={}, eventId={}", taskId, taskNo, eventId);
    }

    public void publishTaskStarted(Long taskId, String eventId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", taskId);
        data.put("eventType", MesEventType.TASK_STARTED.name());
        data.put("eventId", eventId);
        eventPublisher.publish(MesEventType.TASK_STARTED.name(), data);
        log.info("Published TaskStartedEvent: id={}, eventId={}", taskId, eventId);
    }

    public void publishTaskCompleted(Long taskId, Integer qualified, Integer defective, String eventId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", taskId);
        data.put("qualified", qualified);
        data.put("defective", defective);
        data.put("eventType", MesEventType.TASK_COMPLETED.name());
        data.put("eventId", eventId);
        eventPublisher.publish(MesEventType.TASK_COMPLETED.name(), data);
        log.info("Published TaskCompletedEvent: id={}, qualified={}, defective={}, eventId={}", taskId, qualified, defective, eventId);
    }

    public void publishEquipmentBreakdown(Long equipmentId, String eventId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", equipmentId);
        data.put("eventType", MesEventType.EQUIPMENT_BREAKDOWN.name());
        data.put("eventId", eventId);
        eventPublisher.publish(MesEventType.EQUIPMENT_BREAKDOWN.name(), data);
        log.info("Published EquipmentBreakdownEvent: id={}, eventId={}", equipmentId, eventId);
    }

    public void publishEquipmentRepaired(Long equipmentId, String eventId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", equipmentId);
        data.put("eventType", MesEventType.EQUIPMENT_REPAIRED.name());
        data.put("eventId", eventId);
        eventPublisher.publish(MesEventType.EQUIPMENT_REPAIRED.name(), data);
        log.info("Published EquipmentRepairedEvent: id={}, eventId={}", equipmentId, eventId);
    }
}