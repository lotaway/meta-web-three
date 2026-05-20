package com.metawebthree.event;

/**
 * Interface for consuming domain events.
 * Implementations handle event deserialization and processing.
 */
public interface EventConsumer {

    /**
     * Subscribe to events of a specific type.
     *
     * @param eventType     The event type to subscribe to
     * @param handler       The event handler
     * @param <T>           The event type
     */
    <T extends BaseEvent> void subscribe(EventType eventType, EventHandler<T> handler);

    /**
     * Subscribe to all events from a specific topic.
     *
     * @param topic     The topic to subscribe to
     * @param handler   The event handler
     */
    void subscribeToTopic(String topic, EventHandler<BaseEvent> handler);

    /**
     * Start consuming events.
     */
    void start();

    /**
     * Stop consuming events.
     */
    void stop();

    /**
     * Functional interface for handling events.
     *
     * @param <T> The event type
     */
    @FunctionalInterface
    interface EventHandler<T extends BaseEvent> {
        void handle(T event);
    }
}