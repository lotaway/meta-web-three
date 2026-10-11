package com.metawebthree.logistics.infrastructure.event;

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
 * Logistics domain event publisher using Kafka.
 */
@Slf4j
@Component
public class LogisticsDomainEventPublisher {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${event.kafka.topic.prefix:logistics.}")
    private String topicPrefix;

    public LogisticsDomainEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                                          ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    public void publishCreated(String trackingNo, String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("trackingNo", trackingNo);
        data.put("orderNo", orderNo);
        publish("logistics.created", data);
    }

    public void publishTrackingUpdated(String trackingNo, String status, String location) {
        Map<String, Object> data = new HashMap<>();
        data.put("trackingNo", trackingNo);
        data.put("status", status);
        data.put("location", location);
        publish("logistics.tracking.updated", data);
    }

    public void publishDispatched(String trackingNo, String carrier, String carrierOrderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("trackingNo", trackingNo);
        data.put("carrier", carrier);
        data.put("carrierOrderNo", carrierOrderNo);
        publish("logistics.dispatched", data);
    }

    public void publishDelivered(String trackingNo, Long carrierId, String carrierName, 
                                   String orderNo, java.math.BigDecimal freight) {
        Map<String, Object> data = new HashMap<>();
        data.put("trackingNo", trackingNo);
        data.put("orderNo", orderNo);
        data.put("carrierId", carrierId);
        data.put("carrierName", carrierName);
        data.put("freight", freight != null ? freight.toString() : "0");
        publish("logistics.delivered", data);
    }

    private void publish(String eventType, Map<String, Object> data) {
        String topic = topicPrefix + eventType;
        try {
            String message = objectMapper.writeValueAsString(data);
            String key = data.get("trackingNo") != null ? data.get("trackingNo").toString() : null;
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