package com.metawebthree.procurement.infrastructure.event;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Procurement domain event publisher using Kafka.
 */
@Slf4j
@Component
public class ProcurementDomainEventPublisher {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${event.kafka.topic.prefix:procurement.}")
    private String topicPrefix;

    public ProcurementDomainEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                                            ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    public void publishCreated(String orderNo, String supplierCode) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        data.put("supplierCode", supplierCode);
        publish("procurement.created", data);
    }

    public void publishApproved(String orderNo, String approver) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        data.put("approver", approver);
        publish("procurement.approved", data);
    }

    public void publishRejected(String orderNo, String reason) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        data.put("reason", reason);
        publish("procurement.rejected", data);
    }

    public void publishCompleted(String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        publish("procurement.completed", data);
    }

    public void publishCancelled(String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        publish("procurement.cancelled", data);
    }

    private void publish(String eventType, Map<String, Object> data) {
        String topic = topicPrefix + eventType;
        try {
            String message = objectMapper.writeValueAsString(data);
            String key = data.get("orderNo") != null ? data.get("orderNo").toString() : null;
            CompletableFuture<?> future = kafkaTemplate.send(topic, key, message);
            future.whenComplete((result, ex) -> {
                if (ex != null) {
                    log.error("Failed to publish event: topic={}, key={}", topic, key, ex);
                } else {
                    log.debug("Event published: topic={}, key={}", topic, key);
                }
            });
        } catch (JsonProcessingException e) {
            log.error("Failed to serialize event data: topic={}", topic, e);
        }
    }
}