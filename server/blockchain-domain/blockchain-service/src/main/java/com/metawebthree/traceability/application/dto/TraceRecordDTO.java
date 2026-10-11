package com.metawebthree.traceability.application.dto;

import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
public class TraceRecordDTO {

    private Long traceId;

    private String productId;

    private String productName;

    private String batchNumber;

    private String producer;

    private LocalDateTime productionTime;

    private String status;

    private List<TraceEventDTO> events;
}