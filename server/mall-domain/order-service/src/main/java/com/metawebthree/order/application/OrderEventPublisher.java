package com.metawebthree.order.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

/**
 * Order domain event publisher using Kafka.
 */
@Slf4j
@Component
public class OrderEventPublisher {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${event.kafka.topic.order-created:order.created}")
    private String orderCreatedTopic;

    public OrderEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                               ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    /**
     * Publish order created event.
     */
    public void publishOrderCreated(Long orderId, String orderNo, Long userId,
                                     BigDecimal totalAmount, String currency,
                                     List<OrderItemCreate> items) {
        OrderCreatedEvent event = OrderCreatedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.ORDER_CREATED)
                .timestamp(Instant.now())
                .correlationId(orderNo)
                .sourceService("order-service")
                .orderId(orderId.toString())
                .userId(userId.toString())
                .totalAmount(totalAmount)
                .currency(currency)
                .orderTime(Instant.now())
                .items(items.stream()
                        .map(i -> OrderCreatedEvent.OrderItem.builder()
                                .productId(i.getProductId().toString())
                                .productName(i.getProductName())
                                .skuId(i.getSkuId() != null ? i.getSkuId().toString() : null)
                                .quantity(i.getQuantity())
                                .unitPrice(i.getUnitPrice())
                                .imageUrl(i.getImageUrl())
                                .build())
                        .toList())
                .build();

        publish(orderCreatedTopic, orderNo, event);
    }

    private void publish(String topic, String key, Object event) {
        try {
            String payload = objectMapper.writeValueAsString(event);
            CompletableFuture<?> future = kafkaTemplate.send(topic, key, payload);
            future.whenComplete((result, ex) -> {
                if (ex != null) {
                    log.error("Failed to publish event: topic={}, key={}", topic, key, ex);
                } else {
                    log.info("Event published: topic={}, key={}, offset={}",
                            topic, key, result != null ? result.toString() : "N/A");
                }
            });
        } catch (Exception e) {
            log.error("Failed to serialize event: topic={}, key={}", topic, key, e);
        }
    }

    // Event type enum
    public enum EventType {
        ORDER_CREATED("order.created");
        private final String topic;
        EventType(String topic) { this.topic = topic; }
        public String getTopic() { return topic; }
    }

    // Event data class
    public static class OrderCreatedEvent {
        private String eventId;
        private EventType eventType;
        private Instant timestamp;
        private String correlationId;
        private String sourceService;
        private String orderId;
        private String userId;
        private BigDecimal totalAmount;
        private String currency;
        private List<OrderItem> items;
        private Instant orderTime;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final OrderCreatedEvent event = new OrderCreatedEvent();
            public Builder eventId(String val) { event.eventId = val; return this; }
            public Builder eventType(EventType val) { event.eventType = val; return this; }
            public Builder timestamp(Instant val) { event.timestamp = val; return this; }
            public Builder correlationId(String val) { event.correlationId = val; return this; }
            public Builder sourceService(String val) { event.sourceService = val; return this; }
            public Builder orderId(String val) { event.orderId = val; return this; }
            public Builder userId(String val) { event.userId = val; return this; }
            public Builder totalAmount(BigDecimal val) { event.totalAmount = val; return this; }
            public Builder currency(String val) { event.currency = val; return this; }
            public Builder items(List<OrderItem> val) { event.items = val; return this; }
            public Builder orderTime(Instant val) { event.orderTime = val; return this; }
            public OrderCreatedEvent build() { return event; }
        }

        public String getEventId() { return eventId; }
        public EventType getEventType() { return eventType; }
        public Instant getTimestamp() { return timestamp; }
        public String getCorrelationId() { return correlationId; }
        public String getSourceService() { return sourceService; }
        public String getOrderId() { return orderId; }
        public String getUserId() { return userId; }
        public BigDecimal getTotalAmount() { return totalAmount; }
        public String getCurrency() { return currency; }
        public List<OrderItem> getItems() { return items; }
        public Instant getOrderTime() { return orderTime; }

        public static class OrderItem {
            private String productId;
            private String productName;
            private String skuId;
            private Integer quantity;
            private BigDecimal unitPrice;
            private String imageUrl;

            public static Builder builder() { return new Builder(); }
            public static class Builder {
                private final OrderItem item = new OrderItem();
                public Builder productId(String val) { item.productId = val; return this; }
                public Builder productName(String val) { item.productName = val; return this; }
                public Builder skuId(String val) { item.skuId = val; return this; }
                public Builder quantity(Integer val) { item.quantity = val; return this; }
                public Builder unitPrice(BigDecimal val) { item.unitPrice = val; return this; }
                public Builder imageUrl(String val) { item.imageUrl = val; return this; }
                public OrderItem build() { return item; }
            }

            public String getProductId() { return productId; }
            public String getProductName() { return productName; }
            public String getSkuId() { return skuId; }
            public Integer getQuantity() { return quantity; }
            public BigDecimal getUnitPrice() { return unitPrice; }
            public String getImageUrl() { return imageUrl; }
        }
    }

    // Legacy OrderItem for backward compatibility
    public static class OrderItem {
        private String productId;
        private String productName;
        private String skuId;
        private Integer quantity;
        private BigDecimal unitPrice;
        private String imageUrl;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final OrderItem item = new OrderItem();
            public Builder productId(String val) { item.productId = val; return this; }
            public Builder productName(String val) { item.productName = val; return this; }
            public Builder skuId(String val) { item.skuId = val; return this; }
            public Builder quantity(Integer val) { item.quantity = val; return this; }
            public Builder unitPrice(BigDecimal val) { item.unitPrice = val; return this; }
            public Builder imageUrl(String val) { item.imageUrl = val; return this; }
            public OrderItem build() { return item; }
        }

        public String getProductId() { return productId; }
        public String getProductName() { return productName; }
        public String getSkuId() { return skuId; }
        public Integer getQuantity() { return quantity; }
        public BigDecimal getUnitPrice() { return unitPrice; }
        public String getImageUrl() { return imageUrl; }
    }

    // Order item DTO for creation
    public static class OrderItemCreate {
        private Long productId;
        private String productName;
        private Long skuId;
        private Integer quantity;
        private BigDecimal unitPrice;
        private String imageUrl;

        public OrderItemCreate() {}

        public OrderItemCreate(Long productId, String productName, Long skuId, Integer quantity, BigDecimal unitPrice, String imageUrl) {
            this.productId = productId;
            this.productName = productName;
            this.skuId = skuId;
            this.quantity = quantity;
            this.unitPrice = unitPrice;
            this.imageUrl = imageUrl;
        }

        public Long getProductId() { return productId; }
        public void setProductId(Long productId) { this.productId = productId; }
        public String getProductName() { return productName; }
        public void setProductName(String productName) { this.productName = productName; }
        public Long getSkuId() { return skuId; }
        public void setSkuId(Long skuId) { this.skuId = skuId; }
        public Integer getQuantity() { return quantity; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
        public BigDecimal getUnitPrice() { return unitPrice; }
        public void setUnitPrice(BigDecimal unitPrice) { this.unitPrice = unitPrice; }
        public String getImageUrl() { return imageUrl; }
        public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }
    }
}