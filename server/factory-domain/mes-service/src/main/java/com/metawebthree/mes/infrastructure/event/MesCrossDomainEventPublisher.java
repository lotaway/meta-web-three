package com.metawebthree.mes.infrastructure.event;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Cross-domain event publisher for MES -> ERP integration.
 * Publishes standardized events that ERP services can consume.
 */
@Component
public class MesCrossDomainEventPublisher {

    private static final Logger log = LoggerFactory.getLogger(MesCrossDomainEventPublisher.class);

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    public MesCrossDomainEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                                         ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    public void publishWorkOrderCompleted(Long workOrderId, String workOrderNo,
                                           String productCode, Integer quantity) {
        Map<String, Object> data = new HashMap<>();
        data.put("eventId", UUID.randomUUID().toString());
        data.put("eventType", "WORK_ORDER_COMPLETED");
        data.put("timestamp", Instant.now().toString());
        data.put("workOrderId", workOrderId);
        data.put("workOrderNo", workOrderNo);
        data.put("productCode", productCode);
        data.put("quantity", quantity);
        publish("mes.work_order_completed", data);
    }

    public void publishTaskCompleted(Long taskId, String taskNo, Long workOrderId,
                                      String workOrderNo, String productCode,
                                      Integer qualified, Integer defective) {
        Map<String, Object> data = new HashMap<>();
        data.put("eventId", UUID.randomUUID().toString());
        data.put("eventType", "TASK_COMPLETED");
        data.put("timestamp", Instant.now().toString());
        data.put("taskId", taskId);
        data.put("taskNo", taskNo);
        data.put("workOrderId", workOrderId);
        data.put("workOrderNo", workOrderNo);
        data.put("productCode", productCode);
        data.put("qualifiedQuantity", qualified);
        data.put("defectiveQuantity", defective);
        publish("mes.task_completed", data);
    }

    private void publish(String topic, Map<String, Object> data) {
        try {
            String message = objectMapper.writeValueAsString(data);
            String key = data.get("workOrderId") != null
                ? data.get("workOrderId").toString()
                : data.get("taskId").toString();
            kafkaTemplate.send(topic, key, message);
            log.info("Published cross-domain event: topic={}, key={}", topic, key);
        } catch (JsonProcessingException e) {
            log.error("Failed to serialize cross-domain event: topic={}", topic, e);
        }
    }
}
