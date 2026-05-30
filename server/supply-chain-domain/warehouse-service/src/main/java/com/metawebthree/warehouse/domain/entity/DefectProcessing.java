package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 不良品处理记录实体
 * 记录不良品的处理方式
 */
@Data
public class DefectProcessing {
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
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Boolean deleted;

    public static final String TYPE_RETURN = "RETURN";
    public static final String TYPE_EXCHANGE = "EXCHANGE";
    public static final String TYPE_DISCOUNT = "DISCOUNT";
    public static final String TYPE_SCRAP = "SCRAP";
    public static final String TYPE_SPECIAL_USE = "SPECIAL_USE";

    public static final String STATUS_PENDING = "PENDING";
    public static final String STATUS_PROCESSING = "PROCESSING";
    public static final String STATUS_COMPLETED = "COMPLETED";
    public static final String STATUS_CANCELLED = "CANCELLED";

    public boolean isPending() {
        return STATUS_PENDING.equals(this.processingStatus);
    }

    public boolean isCompleted() {
        return STATUS_COMPLETED.equals(this.processingStatus);
    }

    public boolean canCancel() {
        return isPending();
    }

    public void approve(String approver, String remark) {
        this.approver = approver;
        this.approveTime = LocalDateTime.now();
        this.approveRemark = remark;
    }

    public void complete() {
        this.processingStatus = STATUS_COMPLETED;
        this.processingTime = LocalDateTime.now();
    }
}