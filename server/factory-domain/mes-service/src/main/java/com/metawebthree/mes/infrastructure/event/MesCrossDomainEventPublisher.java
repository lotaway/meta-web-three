package com.metawebthree.mes.infrastructure.event;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.event.EventType;
import com.metawebthree.mes.application.event.CrossDomainEventPublisher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@Component
public class MesCrossDomainEventPublisher implements CrossDomainEventPublisher {

    private static final Logger log = LoggerFactory.getLogger(MesCrossDomainEventPublisher.class);

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    public MesCrossDomainEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                                         ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    @Override
    public void publishWorkOrderCompleted(Long workOrderId, String workOrderNo,
                                           String productCode, Integer quantity) {
        Map<String, Object> data = buildEventData(EventType.MES_WORK_ORDER_COMPLETED.name(), workOrderId, workOrderNo, productCode);
        data.put("quantity", quantity);
        publish(EventType.MES_WORK_ORDER_COMPLETED.getTopic(), data);
    }

    @Override
    public void publishTaskCompleted(Long taskId, String taskNo, Long workOrderId,
                                      String workOrderNo, String productCode,
                                      Integer qualified, Integer defective) {
        Map<String, Object> data = buildEventData(EventType.MES_TASK_COMPLETED.name(), workOrderId, workOrderNo, productCode);
        data.put("taskId", taskId);
        data.put("taskNo", taskNo);
        data.put("qualifiedQuantity", qualified);
        data.put("defectiveQuantity", defective);
        publish(EventType.MES_TASK_COMPLETED.getTopic(), data);
    }

    private Map<String, Object> buildEventData(String eventType, Long workOrderId, String workOrderNo, String productCode) {
        Map<String, Object> data = new HashMap<>();
        data.put("eventId", UUID.randomUUID().toString());
        data.put("eventType", eventType);
        data.put("timestamp", Instant.now().toString());
        data.put("workOrderId", workOrderId);
        data.put("workOrderNo", workOrderNo);
        data.put("productCode", productCode);
        return data;
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
            throw new IllegalStateException("Failed to publish cross-domain event: " + topic, e);
        }
    }
}
