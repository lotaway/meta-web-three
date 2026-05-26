package com.metawebthree.supplier.infrastructure.event;

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
 * Supplier domain event publisher using Kafka.
 */
@Slf4j
@Component
public class SupplierDomainEventPublisher {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${event.kafka.topic.prefix:supplier.}")
    private String topicPrefix;

    public SupplierDomainEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                                          ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    public void publishCreated(Long id, String supplierCode, String name) {
        Map<String, Object> data = new HashMap<>();
        data.put("supplierId", id);
        data.put("supplierCode", supplierCode);
        data.put("name", name);
        publish("supplier.created", data);
    }

    public void publishUpdated(Long id, String supplierCode) {
        Map<String, Object> data = new HashMap<>();
        data.put("supplierId", id);
        data.put("supplierCode", supplierCode);
        publish("supplier.updated", data);
    }

    public void publishAssessmentChanged(Long id, String supplierCode, String level) {
        Map<String, Object> data = new HashMap<>();
        data.put("supplierId", id);
        data.put("supplierCode", supplierCode);
        data.put("assessmentLevel", level);
        publish("supplier.assessment.changed", data);
    }

    public void publishStatusChanged(Long id, String supplierCode, String status) {
        Map<String, Object> data = new HashMap<>();
        data.put("supplierId", id);
        data.put("supplierCode", supplierCode);
        data.put("status", status);
        publish("supplier.status.changed", data);
    }

    private void publish(String eventType, Map<String, Object> data) {
        String topic = topicPrefix + eventType;
        try {
            String message = objectMapper.writeValueAsString(data);
            String key = data.get("supplierId") != null ? data.get("supplierId").toString() : null;
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