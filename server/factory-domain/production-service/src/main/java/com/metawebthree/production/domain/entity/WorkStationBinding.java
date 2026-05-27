package com.metawebthree.production.domain.entity;

import java.time.LocalDateTime;

public class WorkStationBinding {
    public static final Integer DEFAULT_QUANTITY = 1;
    public static final Boolean DEFAULT_IS_PRIMARY = false;

    private Long id;
    private String workstationCode;
    private BindingType bindingType;
    private String targetCode;
    private String targetName;
    private String targetType;
    private Integer quantity;
    private Boolean isPrimary;
    private Status status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum BindingType {
        EQUIPMENT, TOOL, PERSONNEL
    }

    public enum Status {
        ACTIVE, INACTIVE
    }

    public void create(String workstationCode, BindingType bindingType, 
                      String targetCode, String targetName, String targetType) {
        this.workstationCode = workstationCode;
        this.bindingType = bindingType;
        this.targetCode = targetCode;
        this.targetName = targetName;
        this.targetType = targetType;
        this.quantity = DEFAULT_QUANTITY;
        this.isPrimary = DEFAULT_IS_PRIMARY;
        this.status = Status.ACTIVE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void bind() {
        if (this.status != Status.INACTIVE) {
            throw new IllegalStateException("Binding is not inactive");
        }
        this.status = Status.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void unbind() {
        if (this.status != Status.ACTIVE) {
            throw new IllegalStateException("Binding is not active");
        }
        this.status = Status.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void setPrimary(Boolean isPrimary) {
        this.isPrimary = isPrimary;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateQuantity(Integer quantity) {
        if (quantity == null || quantity <= 0) {
            throw new IllegalArgumentException("Quantity must be positive");
        }
        this.quantity = quantity;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getWorkstationCode() { return workstationCode; }
    public void setWorkstationCode(String workstationCode) { this.workstationCode = workstationCode; }
    public BindingType getBindingType() { return bindingType; }
    public void setBindingType(BindingType bindingType) { this.bindingType = bindingType; }
    public String getTargetCode() { return targetCode; }
    public void setTargetCode(String targetCode) { this.targetCode = targetCode; }
    public String getTargetName() { return targetName; }
    public void setTargetName(String targetName) { this.targetName = targetName; }
    public String getTargetType() { return targetType; }
    public void setTargetType(String targetType) { this.targetType = targetType; }
    public Integer getQuantity() { return quantity; }
    public void setQuantity(Integer quantity) { this.quantity = quantity; }
    public Boolean getIsPrimary() { return isPrimary; }
    public void setIsPrimary(Boolean isPrimary) { this.isPrimary = isPrimary; }
    public Status getStatus() { return status; }
    public void setStatus(Status status) { this.status = status; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}