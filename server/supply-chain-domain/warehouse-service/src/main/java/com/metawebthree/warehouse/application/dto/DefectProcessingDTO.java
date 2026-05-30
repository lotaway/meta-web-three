package com.metawebthree.warehouse.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class DefectProcessingDTO {
    private Long id;
    private Long defectId;
    private String processingNo;
    private String processingType;
    private String processingStatus;
    private Integer processingQuantity;
    private BigDecimal processingPrice;
    private String processingReason;
    private String processingRemark;
    private String processor;
    private LocalDateTime processingTime;
    private String relatedDocumentNo;
    private String relatedDocumentType;
    private String approver;
    private LocalDateTime approveTime;
    private String approveRemark;
}