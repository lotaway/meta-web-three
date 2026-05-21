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
    @SuperBuilder.Default
    private String eventId = UUID.randomUUID().toString();

    /**
     * Event type, e.g., ORDER_CREATED, INVENTORY_RESERVED.
     */
    private EventType eventType;

    /**
     * Timestamp when event was created.
     */
    @SuperBuilder.Default
    private Instant timestamp = Instant.now();

    /**
     * Correlation ID for tracking related events.
     */
    private String correlationId;

    /**
     * Source service that published this event.
     */
    private String sourceService;

    /**
     * Payload version for schema evolution.
     */
    @SuperBuilder.Default
    private int version = 1;
}