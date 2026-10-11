package com.metawebthree.traceability.application.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class TraceEventDTO {

    private Long traceId;

    private String eventType;

    private String description;

    private String location;

    private String operator;

    private LocalDateTime timestamp;

    private String extraData;
}