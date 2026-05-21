package com.metawebthree.event;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Kafka implementation of EventConsumer.
 */
@Slf4j
@Component
public class KafkaEventConsumer implements EventConsumer {

    private final ObjectMapper objectMapper;
    private final Map<String, EventHandler<BaseEvent>> handlers = new HashMap<>();
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private boolean started = false;

    public KafkaEventConsumer(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @Override
    public <T extends BaseEvent> void subscribe(EventType eventType, EventHandler<T> handler) {
        @SuppressWarnings("unchecked")
        EventHandler<BaseEvent> castHandler = event -> handler.handle((T) event);
        handlers.put(eventType.getTopic(), castHandler);
    }

    @Override
    public void subscribeToTopic(String topic, EventHandler<BaseEvent> handler) {
        handlers.put(topic, handler);
    }

    @Override
    public void start() {
        if (!started) {
            started = true;
            log.info("Event consumer started, listening to {} topics", handlers.size());
        }
    }

    @Override
    public void stop() {
        if (started) {
            started = false;
            executor.shutdown();
            log.info("Event consumer stopped");
        }
    }

    @PostConstruct
    public void init() {
        start();
    }

    @PreDestroy
    public void shutdown() {
        stop();
    }

    /**
     * Kafka listener for all registered topics.
     * Uses pattern matching to listen to multiple topics.
     */
    @KafkaListener(topics = "${event.consumer.topics:order.created,inventory.reserved,payment.completed,shipment.created,inventory.released}", 
                   groupId = "${spring.kafka.consumer.group-id:meta-web-three}")
    public void consume(String message) {
        try {
            BaseEvent event = objectMapper.readValue(message, BaseEvent.class);
            String topic = event.getEventType().getTopic();

            EventHandler<BaseEvent> handler = handlers.get(topic);
            if (handler != null) {
                executor.submit(() -> handler.handle(event));
            } else {
                log.debug("No handler registered for topic: {}, skipping", topic);
            }
        } catch (JsonProcessingException e) {
            log.error("Failed to deserialize event message: {}", message, e);
        }
    }

    /**
     * Get all registered topic patterns.
     */
    public String topicPattern() {
        return String.join(",", handlers.keySet());
    }
}