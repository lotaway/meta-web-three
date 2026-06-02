package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class StocktakeOrder {
    public static final String STATUS_DRAFT = "DRAFT";
    public static final String STATUS_PENDING = "PENDING";
    public static final String STATUS_IN_PROGRESS = "IN_PROGRESS";
    public static final String STATUS_COUNTING = "COUNTING";
    public static final String STATUS_DISCREPANCY_REPORTED = "DISCREPANCY_REPORTED";
    public static final String STATUS_ADJUSTED = "ADJUSTED";
    public static final String STATUS_COMPLETED = "COMPLETED";
    public static final String STATUS_CANCELLED = "CANCELLED";

    public static final String TYPE_CYCLE = "CYCLE";
    public static final String TYPE_FULL = "FULL";
    public static final String TYPE_SPOT = "SPOT";

    private Long id;
    private String orderNo;
    private String type;
    private Long warehouseId;
    private String warehouseName;
    private Long locationId;
    private String locationName;
    private String status;
    private String operator;
    private LocalDateTime plannedDate;
    private LocalDateTime startDate;
    private LocalDateTime endDate;
    private Integer totalSkuCount;
    private Integer checkedSkuCount;
    private Integer discrepancyCount;
    private BigDecimal totalDiscrepancyAmount;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<StocktakeOrderItem> items;

    public void submit() {
        if (STATUS_DRAFT.equals(this.status)) {
            this.status = STATUS_PENDING;
        }
    }

    public void start() {
        if (STATUS_PENDING.equals(this.status)) {
            this.status = STATUS_IN_PROGRESS;
            this.startDate = LocalDateTime.now();
        }
    }

    public void completeCounting() {
        if (STATUS_IN_PROGRESS.equals(this.status) || STATUS_COUNTING.equals(this.status)) {
            this.status = STATUS_COUNTING;
        }
    }

    public void reportDiscrepancy() {
        if (STATUS_COUNTING.equals(this.status)) {
            this.status = STATUS_DISCREPANCY_REPORTED;
            this.endDate = LocalDateTime.now();
        }
    }

    public void adjustInventory() {
        if (STATUS_DISCREPANCY_REPORTED.equals(this.status)) {
            this.status = STATUS_ADJUSTED;
        }
    }

    public void complete() {
        if (STATUS_ADJUSTED.equals(this.status) || STATUS_COUNTING.equals(this.status)) {
            this.status = STATUS_COMPLETED;
            this.endDate = LocalDateTime.now();
        }
    }

    public void cancel() {
        if (STATUS_DRAFT.equals(this.status) || STATUS_PENDING.equals(this.status)) {
            this.status = STATUS_CANCELLED;
        }
    }
}