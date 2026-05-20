package com.metawebthree.event;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/**
 * Kafka implementation of EventPublisher.
 */
@Slf4j
@Component
public class KafkaEventPublisher implements EventPublisher {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    public KafkaEventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                               ObjectMapper objectMapper) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    @Override
    public <T extends BaseEvent> void publish(T event) {
        publish(event, event.getCorrelationId());
    }

    @Override
    public <T extends BaseEvent> void publish(T event, String correlationId) {
        String topic = event.getEventType().getTopic();
        String key = correlationId != null ? correlationId : event.getEventId();

        try {
            String payload = objectMapper.writeValueAsString(event);
            CompletableFuture<SendResult<String, String>> future =
                kafkaTemplate.send(topic, key, payload);

            future.whenComplete((result, ex) -> {
                if (ex != null) {
                    log.error("Failed to publish event: type={}, id={}",
                        event.getEventType(), event.getEventId(), ex);
                } else {
                    log.debug("Event published: type={}, id={}, partition={}, offset={}",
                        event.getEventType(), event.getEventId(),
                        result.getRecordMetadata().partition(),
                        result.getRecordMetadata().offset());
                }
            });
        } catch (JsonProcessingException e) {
            log.error("Failed to serialize event: type={}, id={}",
                event.getEventType(), event.getEventId(), e);
            throw new RuntimeException("Event serialization failed", e);
        }
    }
}