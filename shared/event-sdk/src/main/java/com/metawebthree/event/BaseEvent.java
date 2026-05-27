package com.metawebthree.event;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.time.Instant;
import java.util.UUID;

/**
 * Base event model for all domain events.
 * All events should extend this class.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
@JsonIgnoreProperties(ignoreUnknown = true)
public class BaseEvent {

    /**
     * Unique event identifier.
     */
    protected String eventId = UUID.randomUUID().toString();

    /**
     * Event type, e.g., ORDER_CREATED, INVENTORY_RESERVED.
     */
    protected EventType eventType;

    /**
     * Timestamp when event was created.
     */
    protected Instant timestamp = Instant.now();

    /**
     * Correlation ID for tracking related events.
     */
    protected String correlationId;

    /**
     * Source service that published this event.
     */
    protected String sourceService;

    /**
     * Payload version for schema evolution.
     */
    protected int version = 1;
}