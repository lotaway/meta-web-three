package com.metawebthree.procurement.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class ProcurementReturnOrder {
    
    public static final String STATUS_DRAFT = "DRAFT";
    public static final String STATUS_PENDING_APPROVAL = "PENDING_APPROVAL";
    public static final String STATUS_APPROVED = "APPROVED";
    public static final String STATUS_REJECTED = "REJECTED";
    public static final String STATUS_RETURNING = "RETURNING";
    public static final String STATUS_RETURNED = "RETURNED";
    public static final String STATUS_COMPLETED = "COMPLETED";
    public static final String STATUS_CANCELLED = "CANCELLED";
    
    public static final String RETURN_TYPE_QUALITY = "QUALITY_ISSUE";
    public static final String RETURN_TYPE_WRONG = "WRONG_DELIVERY";
    public static final String RETURN_TYPE_OVERSTOCK = "OVERSTOCK";
    public static final String RETURN_TYPE_OTHER = "OTHER";
    
    private Long id;
    private String returnNo;
    private String sourceOrderNo;
    private String sourceOrderType;
    private String supplierCode;
    private String supplierName;
    private Long warehouseId;
    private String warehouseName;
    private String returnType;
    private String status;
    private BigDecimal totalAmount;
    private String currency;
    private String reason;
    private String remark;
    private String approver;
    private String approvalComment;
    private LocalDateTime approvedAt;
    private LocalDateTime expectedReturnDate;
    private LocalDateTime actualReturnDate;
    private String logisticsCompany;
    private String trackingNumber;
    private LocalDateTime shippedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private List<ProcurementReturnOrderItem> items;
    
    public void submitForApproval() {
        if (STATUS_DRAFT.equals(this.status)) {
            this.status = STATUS_PENDING_APPROVAL;
        }
    }
    
    public void approve(String approver, String comment) {
        if (STATUS_PENDING_APPROVAL.equals(this.status)) {
            this.status = STATUS_APPROVED;
            this.approver = approver;
            this.approvalComment = comment;
            this.approvedAt = LocalDateTime.now();
        }
    }
    
    public void reject(String approver, String reason) {
        if (STATUS_PENDING_APPROVAL.equals(this.status)) {
            this.status = STATUS_REJECTED;
            this.approver = approver;
            this.approvalComment = reason;
            this.approvedAt = LocalDateTime.now();
        }
    }
    
    public void ship(String logisticsCompany, String trackingNumber) {
        if (STATUS_APPROVED.equals(this.status)) {
            this.status = STATUS_RETURNING;
            this.logisticsCompany = logisticsCompany;
            this.trackingNumber = trackingNumber;
            this.shippedAt = LocalDateTime.now();
        }
    }
    
    public void confirmReturned() {
        if (STATUS_RETURNING.equals(this.status)) {
            this.status = STATUS_RETURNED;
            this.actualReturnDate = LocalDateTime.now();
        }
    }
    
    public void complete() {
        if (STATUS_RETURNED.equals(this.status)) {
            this.status = STATUS_COMPLETED;
        }
    }
    
    public void cancel() {
        if (STATUS_DRAFT.equals(this.status) || STATUS_PENDING_APPROVAL.equals(this.status)) {
            this.status = STATUS_CANCELLED;
        }
    }
}