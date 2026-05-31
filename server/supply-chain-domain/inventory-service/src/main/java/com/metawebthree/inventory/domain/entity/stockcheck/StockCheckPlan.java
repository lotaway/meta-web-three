package com.metawebthree.inventory.domain.entity.stockcheck;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 盘点计划实体
 */
@Data
public class StockCheckPlan {
    private Long id;
    private String planNo;
    private String planName;
    private String checkType;
    private Long warehouseId;
    private String warehouseName;
    private String status;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private String remark;
    private Boolean deleted;
    private Integer version;

    private List<StockCheckPlanDetail> details;

    public static final String TYPE_FULL = "FULL";
    public static final String TYPE_SPOT = "SPOT";
    public static final String TYPE_CYCLE = "CYCLE";

    public static final String STATUS_DRAFT = "DRAFT";
    public static final String STATUS_PENDING = "PENDING";
    public static final String STATUS_IN_PROGRESS = "IN_PROGRESS";
    public static final String STATUS_COMPLETED = "COMPLETED";
    public static final String STATUS_CANCELLED = "CANCELLED";

    public void approve() {
        if (!STATUS_DRAFT.equals(this.status) && !STATUS_PENDING.equals(this.status)) {
            throw new IllegalStateException("Only draft or pending plan can be approved");
        }
        this.status = STATUS_PENDING;
    }

    public void start() {
        if (!STATUS_PENDING.equals(this.status)) {
            throw new IllegalStateException("Only pending plan can be started");
        }
        this.status = STATUS_IN_PROGRESS;
        this.actualStartTime = LocalDateTime.now();
    }

    public void complete() {
        if (!STATUS_IN_PROGRESS.equals(this.status)) {
            throw new IllegalStateException("Only in-progress plan can be completed");
        }
        this.status = STATUS_COMPLETED;
        this.actualEndTime = LocalDateTime.now();
    }

    public void cancel() {
        if (STATUS_COMPLETED.equals(this.status) || STATUS_CANCELLED.equals(this.status)) {
            throw new IllegalStateException("Completed or cancelled plan cannot be cancelled");
        }
        this.status = STATUS_CANCELLED;
    }

    public boolean isEditable() {
        return STATUS_DRAFT.equals(this.status);
    }
}