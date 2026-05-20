package com.metawebthree.mes.infrastructure.event;

import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class MesEventPublisher {

    public void publishWorkOrderCreated(Long workOrderId, String workOrderNo) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORK_ORDER_CREATED");
        event.put("workOrderId", workOrderId);
        event.put("workOrderNo", workOrderNo);
    }

    public void publishWorkOrderReleased(Long workOrderId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORK_ORDER_RELEASED");
        event.put("workOrderId", workOrderId);
    }

    public void publishWorkOrderStarted(Long workOrderId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORK_ORDER_STARTED");
        event.put("workOrderId", workOrderId);
    }

    public void publishWorkOrderCompleted(Long workOrderId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "WORK_ORDER_COMPLETED");
        event.put("workOrderId", workOrderId);
    }

    public void publishTaskCreated(Long taskId, String taskNo) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "TASK_CREATED");
        event.put("taskId", taskId);
        event.put("taskNo", taskNo);
    }

    public void publishTaskStarted(Long taskId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "TASK_STARTED");
        event.put("taskId", taskId);
    }

    public void publishTaskCompleted(Long taskId, Integer qualified, Integer defective) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "TASK_COMPLETED");
        event.put("taskId", taskId);
        event.put("qualifiedQuantity", qualified);
        event.put("defectiveQuantity", defective);
    }

    public void publishEquipmentBreakdown(Long equipmentId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "EQUIPMENT_BREAKDOWN");
        event.put("equipmentId", equipmentId);
    }

    public void publishEquipmentRepaired(Long equipmentId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "EQUIPMENT_REPAIRED");
        event.put("equipmentId", equipmentId);
    }
}