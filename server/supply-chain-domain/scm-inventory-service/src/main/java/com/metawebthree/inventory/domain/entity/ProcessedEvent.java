package com.metawebthree.inventory.domain.entity;

import lombok.Data;

import java.time.LocalDateTime;

/**
 * Records a domain event that a consumer has already handled, providing
 * de-duplication for at-least-once delivery.
 */
@Data
public class ProcessedEvent {
    private Long id;
    private String eventId;
    private String consumerGroup;
    private LocalDateTime processedAt;
}
