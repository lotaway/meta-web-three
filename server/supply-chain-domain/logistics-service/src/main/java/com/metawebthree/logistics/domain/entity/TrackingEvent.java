package com.metawebthree.logistics.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class TrackingEvent {
    private Long id;
    private String trackingNo;
    private String eventType;
    private String location;
    private String description;
    private String operator;
    private LocalDateTime occurredAt;
    private LocalDateTime createdAt;
}