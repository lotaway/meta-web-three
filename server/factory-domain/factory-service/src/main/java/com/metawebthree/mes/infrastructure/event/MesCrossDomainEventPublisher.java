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
        WorkOrderCompletedEvent event = new WorkOrderCompletedEvent(
            UUID.randomUUID().toString(),
            EventType.MES_WORK_ORDER_COMPLETED.name(),
            Instant.now().toString(),
            workOrderId, workOrderNo, productCode, quantity
        );
        publish(EventType.MES_WORK_ORDER_COMPLETED.getTopic(), event);
    }

    @Override
    public void publishTaskCompleted(Long taskId, String taskNo, Long workOrderId,
                                      String workOrderNo, String productCode,
                                      Integer qualified, Integer defective) {
        TaskCompletedEvent event = new TaskCompletedEvent(
            UUID.randomUUID().toString(),
            EventType.MES_TASK_COMPLETED.name(),
            Instant.now().toString(),
            workOrderId, workOrderNo, productCode,
            taskId, taskNo, qualified, defective
        );
        publish(EventType.MES_TASK_COMPLETED.getTopic(), event);
    }

    private void publish(String topic, Object event) {
        try {
            String message = objectMapper.writeValueAsString(event);
            String key;
            if (event instanceof WorkOrderCompletedEvent w) {
                key = w.workOrderId().toString();
            } else if (event instanceof TaskCompletedEvent t) {
                key = t.taskId().toString();
            } else {
                key = UUID.randomUUID().toString();
            }
            kafkaTemplate.send(topic, key, message);
            log.info("Published cross-domain event: topic={}, key={}", topic, key);
        } catch (JsonProcessingException e) {
            log.error("Failed to serialize cross-domain event: topic={}", topic, e);
            throw new IllegalStateException("Failed to publish cross-domain event: " + topic, e);
        }
    }

    private record WorkOrderCompletedEvent(String eventId, String eventType, String timestamp,
                                            Long workOrderId, String workOrderNo, String productCode,
                                            Integer quantity) {}

    private record TaskCompletedEvent(String eventId, String eventType, String timestamp,
                                       Long workOrderId, String workOrderNo, String productCode,
                                       Long taskId, String taskNo,
                                       Integer qualifiedQuantity, Integer defectiveQuantity) {}
}
