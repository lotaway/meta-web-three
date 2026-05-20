package com.metawebthree.event;

/**
 * Interface for publishing domain events.
 * Implementations can use different message brokers (Kafka, RabbitMQ, etc.).
 */
public interface EventPublisher {

    /**
     * Publish a domain event to the event bus.
     *
     * @param event The event to publish
     * @param <T>   The event type
     */
    <T extends BaseEvent> void publish(T event);

    /**
     * Publish a domain event with a specific correlation ID.
     *
     * @param event          The event to publish
     * @param correlationId  The correlation ID for tracking
     * @param <T>            The event type
     */
    <T extends BaseEvent> void publish(T event, String correlationId);
}