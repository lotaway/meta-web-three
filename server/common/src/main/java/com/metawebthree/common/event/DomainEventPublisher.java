package com.metawebthree.common.event;

import java.util.Map;

/**
 * Event publisher interface for domain events.
 */
public interface DomainEventPublisher {

    /**
     * Publish an event with data.
     * @param eventType event type
     * @param data event data
     */
    void publish(String eventType, Map<String, Object> data);
}