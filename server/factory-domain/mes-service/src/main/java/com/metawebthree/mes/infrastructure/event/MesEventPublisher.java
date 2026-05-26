package com.metawebthree.mes.infrastructure.event;

import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class MesEventPublisher {

    private final ApplicationEventPublisher eventPublisher;

    public MesEventPublisher(ApplicationEventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishWorkOrderCreated(Long workOrderId, String workOrderNo) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORK_ORDER_CREATED");
        event.put("workOrderId", workOrderId);
        event.put("workOrderNo", workOrderNo);
        eventPublisher.publishEvent(event);
    }

    public void publishWorkOrderReleased(Long workOrderId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORK_ORDER_RELEASED");
        event.put("workOrderId", workOrderId);
        eventPublisher.publishEvent(event);
    }

    public void publishWorkOrderStarted(Long workOrderId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORK_ORDER_STARTED");
        event.put("workOrderId", workOrderId);
        eventPublisher.publishEvent(event);
    }

    public void publishWorkOrderCompleted(Long workOrderId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORK_ORDER_COMPLETED");
        event.put("workOrderId", workOrderId);
        eventPublisher.publishEvent(event);
    }

    public void publishTaskCreated(Long taskId, String taskNo) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "TASK_CREATED");
        event.put("taskId", taskId);
        event.put("taskNo", taskNo);
        eventPublisher.publishEvent(event);
    }

    public void publishTaskStarted(Long taskId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "TASK_STARTED");
        event.put("taskId", taskId);
        eventPublisher.publishEvent(event);
    }

    public void publishTaskCompleted(Long taskId, Integer qualified, Integer defective) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "TASK_COMPLETED");
        event.put("taskId", taskId);
        event.put("qualifiedQuantity", qualified);
        event.put("defectiveQuantity", defective);
        eventPublisher.publishEvent(event);
    }

    public void publishEquipmentBreakdown(Long equipmentId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "EQUIPMENT_BREAKDOWN");
        event.put("equipmentId", equipmentId);
        eventPublisher.publishEvent(event);
    }

    public void publishEquipmentRepaired(Long equipmentId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "EQUIPMENT_REPAIRED");
        event.put("equipmentId", equipmentId);
        eventPublisher.publishEvent(event);
    }
}