package com.metawebthree.inventory.domain.entity.stockcheck;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 盘点差异处理记录
 */
@Data
@TableName("tb_stock_check_diff")
public class StockCheckDiff {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long recordId;
    private Long planId;
    private String planNo;
    private String skuCode;
    private String productName;
    private String locationCode;
    private Long warehouseId;
    private BigDecimal bookQuantity;
    private BigDecimal checkQuantity;
    private BigDecimal differenceQuantity;
    private String differenceType;
    private String processingStatus;
    private String approvalStatus;
    private String approver;
    private LocalDateTime approvalTime;
    private String approvalRemark;
    private String solution;
    private String processor;
    private LocalDateTime processTime;
    private String processRemark;
    private String sourceSystem;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Boolean deleted;
    private Integer version;

    public static final String DIFF_TYPE_SHORT = "SHORT";
    public static final String DIFF_TYPE_OVER = "OVER";

    public static final String PROCESS_STATUS_PENDING = "PENDING";
    public static final String PROCESS_STATUS_PROCESSING = "PROCESSING";
    public static final String PROCESS_STATUS_PROCESSED = "PROCESSED";

    public static final String APPROVAL_STATUS_PENDING = "PENDING";
    public static final String APPROVAL_STATUS_APPROVED = "APPROVED";
    public static final String APPROVAL_STATUS_REJECTED = "REJECTED";

    public boolean needsApproval() {
        return differenceQuantity != null && 
               differenceQuantity.abs().compareTo(BigDecimal.TEN) > 0;
    }

    public void approve(String approver, String remark) {
        if (!APPROVAL_STATUS_PENDING.equals(this.approvalStatus)) {
            throw new IllegalStateException("Only pending diff can be approved");
        }
        this.approvalStatus = APPROVAL_STATUS_APPROVED;
        this.approver = approver;
        this.approvalTime = LocalDateTime.now();
        this.approvalRemark = remark;
    }

    public void reject(String approver, String remark) {
        if (!APPROVAL_STATUS_PENDING.equals(this.approvalStatus)) {
            throw new IllegalStateException("Only pending diff can be rejected");
        }
        this.approvalStatus = APPROVAL_STATUS_REJECTED;
        this.approver = approver;
        this.approvalTime = LocalDateTime.now();
        this.approvalRemark = remark;
    }

    public void process(String processor, String solution, String remark) {
        if (!APPROVAL_STATUS_APPROVED.equals(this.approvalStatus)) {
            throw new IllegalStateException("Only approved diff can be processed");
        }
        this.processingStatus = PROCESS_STATUS_PROCESSED;
        this.processor = processor;
        this.solution = solution;
        this.processTime = LocalDateTime.now();
        this.processRemark = remark;
    }
}