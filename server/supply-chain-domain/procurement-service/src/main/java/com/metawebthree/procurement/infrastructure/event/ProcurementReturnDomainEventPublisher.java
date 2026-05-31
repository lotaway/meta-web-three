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
 * Procurement return order domain event publisher using Kafka.
 */
@Slf4j
@Component
public class ProcurementReturnDomainEventPublisher {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${event.kafka.topic.prefix:procurement.}")
    private String topicPrefix;

    public ProcurementReturnDomainEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                                                  ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    public void publishReturnOrderCreated(String returnNo, String sourceOrderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("returnNo", returnNo);
        data.put("sourceOrderNo", sourceOrderNo);
        publish("return.created", data);
    }

    public void publishReturnOrderSubmitted(String returnNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("returnNo", returnNo);
        publish("return.submitted", data);
    }

    public void publishReturnOrderApproved(String returnNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("returnNo", returnNo);
        publish("return.approved", data);
    }

    public void publishReturnOrderRejected(String returnNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("returnNo", returnNo);
        publish("return.rejected", data);
    }

    public void publishReturnOrderShipped(String returnNo, String logisticsCompany, String trackingNumber) {
        Map<String, Object> data = new HashMap<>();
        data.put("returnNo", returnNo);
        data.put("logisticsCompany", logisticsCompany);
        data.put("trackingNumber", trackingNumber);
        publish("return.shipped", data);
    }

    public void publishReturnOrderConfirmed(String returnNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("returnNo", returnNo);
        publish("return.confirmed", data);
    }

    public void publishReturnOrderCompleted(String returnNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("returnNo", returnNo);
        publish("return.completed", data);
    }

    public void publishReturnOrderCancelled(String returnNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("returnNo", returnNo);
        publish("return.cancelled", data);
    }

    private void publish(String eventType, Map<String, Object> data) {
        String topic = topicPrefix + eventType;
        try {
            String message = objectMapper.writeValueAsString(data);
            String key = data.get("returnNo") != null ? data.get("returnNo").toString() : null;
            CompletableFuture<?> future = kafkaTemplate.send(topic, key, message);
            future.whenComplete((result, ex) -> {
                if (ex != null) {
                    log.error("Failed to publish return event: topic={}, key={}", topic, key, ex);
                } else {
                    log.debug("Return event published: topic={}, key={}", topic, key);
                }
            });
        } catch (JsonProcessingException e) {
            log.error("Failed to serialize return event data: topic={}", topic, e);
        }
    }
}