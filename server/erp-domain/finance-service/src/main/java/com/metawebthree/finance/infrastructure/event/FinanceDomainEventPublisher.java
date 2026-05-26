package com.metawebthree.finance.infrastructure.event;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

/**
 * Finance domain event publisher using Kafka.
 */
@Slf4j
@Component
public class FinanceDomainEventPublisher {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${event.kafka.topic.prefix:finance.}")
    private String topicPrefix;

    public FinanceDomainEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                                         ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    public void publishAccountCreated(Long accountId, String accountNo, String accountName, String accountType) {
        Map<String, Object> data = new HashMap<>();
        data.put("eventId", UUID.randomUUID().toString());
        data.put("eventType", "ACCOUNT_CREATED");
        data.put("timestamp", Instant.now().toString());
        data.put("accountId", accountId);
        data.put("accountNo", accountNo);
        data.put("accountName", accountName);
        data.put("accountType", accountType);
        publish("account.created", data);
    }

    public void publishAccountBalanceChanged(Long accountId, String accountNo,
            BigDecimal beforeBalance, BigDecimal afterBalance, String changeType) {
        Map<String, Object> data = new HashMap<>();
        data.put("eventId", UUID.randomUUID().toString());
        data.put("eventType", "BALANCE_CHANGED");
        data.put("timestamp", Instant.now().toString());
        data.put("accountId", accountId);
        data.put("accountNo", accountNo);
        data.put("beforeBalance", beforeBalance);
        data.put("afterBalance", afterBalance);
        data.put("changeType", changeType);
        publish("balance.changed", data);
    }

    private void publish(String eventType, Map<String, Object> data) {
        String topic = topicPrefix + eventType;
        try {
            String message = objectMapper.writeValueAsString(data);
            String key = data.get("accountId") != null ? data.get("accountId").toString() : null;
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