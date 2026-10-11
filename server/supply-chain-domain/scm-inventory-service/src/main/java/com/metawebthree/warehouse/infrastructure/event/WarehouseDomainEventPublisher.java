package com.metawebthree.warehouse.infrastructure.event;

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
 * Warehouse domain event publisher using Kafka.
 */
@Slf4j
@Component
public class WarehouseDomainEventPublisher {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${event.kafka.topic.prefix:warehouse.}")
    private String topicPrefix;

    public WarehouseDomainEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                                          ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    public void publishCreated(Long warehouseId, String warehouseCode, String name) {
        Map<String, Object> data = new HashMap<>();
        data.put("warehouseId", warehouseId);
        data.put("warehouseCode", warehouseCode);
        data.put("name", name);
        publish("warehouse.created", data);
    }

    public void publishStockIn(Long warehouseId, String skuCode, Integer quantity, String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("warehouseId", warehouseId);
        data.put("skuCode", skuCode);
        data.put("quantity", quantity);
        data.put("orderNo", orderNo);
        publish("warehouse.stock.in", data);
    }

    public void publishStockOut(Long warehouseId, String skuCode, Integer quantity, String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("warehouseId", warehouseId);
        data.put("skuCode", skuCode);
        data.put("quantity", quantity);
        data.put("orderNo", orderNo);
        publish("warehouse.stock.out", data);
    }

    public void publishInboundOrderCreated(String orderNo, Long warehouseId) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        data.put("warehouseId", warehouseId);
        publish("inbound.order.created", data);
    }

    public void publishInboundOrderCompleted(String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        publish("inbound.order.completed", data);
    }

    private void publish(String eventType, Map<String, Object> data) {
        String topic = topicPrefix + eventType;
        try {
            String message = objectMapper.writeValueAsString(data);
            String key = data.get("warehouseId") != null ? data.get("warehouseId").toString() : null;
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