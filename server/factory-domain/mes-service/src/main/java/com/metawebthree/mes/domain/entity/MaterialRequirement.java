package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class MaterialRequirement {
    
    private static final double DEFAULT_SCRAP_RATE = 0.01;
    
    private Long id;
    private String requirementNo;
    private String workOrderNo;
    private String productCode;
    private String productName;
    private Integer quantity;
    private String bomVersion;
    private String status;
    private String warehouseId;
    private String workshopId;
    private String requirementType;
    private LocalDateTime requiredDate;
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private List<MaterialRequirementItem> items = new ArrayList<>();
    
    public enum RequirementStatus {
        DRAFT,
        CONFIRMED,
        ISSUED,
        PARTIALLY_ISSUED,
        COMPLETED
    }
    
    public enum RequirementType {
        NORMAL,
        EMERGENCY,
        REWORK
    }
    
    public static class MaterialRequirementItem {
        private Long id;
        private Long requirementId;
        private String materialCode;
        private String materialName;
        private String materialSpec;
        private String unitCode;
        private String unitName;
        private Double requiredQuantity;
        private Double issuedQuantity;
        private Double pendingQuantity;
        private Double scrapQuantity;
        private String locationId;
        private String batchNo;
        private String status;
        private String remark;
        private LocalDateTime requiredDate;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;
        
        public enum ItemStatus {
            PENDING,
            ISSUED,
            PARTIALLY_ISSUED
        }
        
        public void create(Long requirementId, String materialCode, String materialName,
                          Double requiredQuantity, String unitCode, String unitName) {
            this.requirementId = requirementId;
            this.materialCode = materialCode;
            this.materialName = materialName;
            this.requiredQuantity = requiredQuantity;
            this.issuedQuantity = 0.0;
            this.pendingQuantity = requiredQuantity;
            this.scrapQuantity = 0.0;
            this.status = ItemStatus.PENDING.name();
            this.createdAt = LocalDateTime.now();
            this.updatedAt = LocalDateTime.now();
        }
        
        public void issue(Double quantity) {
            if (quantity <= 0) {
                throw new IllegalArgumentException("Issue quantity must be positive");
            }
            if (quantity > pendingQuantity) {
                throw new IllegalArgumentException("Issue quantity exceeds pending quantity");
            }
            this.issuedQuantity += quantity;
            this.pendingQuantity -= quantity;
            this.scrapQuantity += quantity * DEFAULT_SCRAP_RATE;
            
            // 更新状态
            if (pendingQuantity <= 0) {
                this.status = ItemStatus.ISSUED.name();
            } else {
                this.status = ItemStatus.PARTIALLY_ISSUED.name();
            }
            this.updatedAt = LocalDateTime.now();
        }
        
        public void cancelIssue(Double quantity) {
            if (quantity <= 0) {
                throw new IllegalArgumentException("Cancel quantity must be positive");
            }
            if (quantity > issuedQuantity) {
                throw new IllegalArgumentException("Cancel quantity exceeds issued quantity");
            }
            this.issuedQuantity -= quantity;
            this.pendingQuantity += quantity;
            this.scrapQuantity = Math.max(0, this.scrapQuantity - quantity * DEFAULT_SCRAP_RATE);
            
            // 更新状态
            if (issuedQuantity <= 0) {
                this.status = ItemStatus.PENDING.name();
            } else {
                this.status = ItemStatus.PARTIALLY_ISSUED.name();
            }
            this.updatedAt = LocalDateTime.now();
        }
        
        public void complete() {
            this.status = ItemStatus.ISSUED.name();
            this.pendingQuantity = 0.0;
            this.updatedAt = LocalDateTime.now();
        }
        
        public Double getIssueCompletionRate() {
            if (requiredQuantity == null || requiredQuantity == 0) {
                return 0.0;
            }
            return (issuedQuantity / requiredQuantity) * 100;
        }
        
        // Getters and Setters
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public Long getRequirementId() { return requirementId; }
        public void setRequirementId(Long requirementId) { this.requirementId = requirementId; }
        public String getMaterialCode() { return materialCode; }
        public void setMaterialCode(String materialCode) { this.materialCode = materialCode; }
        public String getMaterialName() { return materialName; }
        public void setMaterialName(String materialName) { this.materialName = materialName; }
        public String getMaterialSpec() { return materialSpec; }
        public void setMaterialSpec(String materialSpec) { this.materialSpec = materialSpec; }
        public String getUnitCode() { return unitCode; }
        public void setUnitCode(String unitCode) { this.unitCode = unitCode; }
        public String getUnitName() { return unitName; }
        public void setUnitName(String unitName) { this.unitName = unitName; }
        public Double getRequiredQuantity() { return requiredQuantity; }
        public void setRequiredQuantity(Double requiredQuantity) { this.requiredQuantity = requiredQuantity; }
        public Double getIssuedQuantity() { return issuedQuantity; }
        public void setIssuedQuantity(Double issuedQuantity) { this.issuedQuantity = issuedQuantity; }
        public Double getPendingQuantity() { return pendingQuantity; }
        public void setPendingQuantity(Double pendingQuantity) { this.pendingQuantity = pendingQuantity; }
        public Double getScrapQuantity() { return scrapQuantity; }
        public void setScrapQuantity(Double scrapQuantity) { this.scrapQuantity = scrapQuantity; }
        public String getLocationId() { return locationId; }
        public void setLocationId(String locationId) { this.locationId = locationId; }
        public String getBatchNo() { return batchNo; }
        public void setBatchNo(String batchNo) { this.batchNo = batchNo; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getRemark() { return remark; }
        public void setRemark(String remark) { this.remark = remark; }
        public LocalDateTime getRequiredDate() { return requiredDate; }
        public void setRequiredDate(LocalDateTime requiredDate) { this.requiredDate = requiredDate; }
        public LocalDateTime getCreatedAt() { return createdAt; }
        public LocalDateTime getUpdatedAt() { return updatedAt; }
    }
    
    public void create(String requirementNo, String workOrderNo, String productCode,
                      String productName, Integer quantity) {
        this.requirementNo = requirementNo;
        this.workOrderNo = workOrderNo;
        this.productCode = productCode;
        this.productName = productName;
        this.quantity = quantity;
        this.status = RequirementStatus.DRAFT.name();
        this.requirementType = RequirementType.NORMAL.name();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void addItem(MaterialRequirementItem item) {
        // 检查物料是否重复
        boolean exists = items.stream()
                .anyMatch(i -> i.getMaterialCode().equals(item.getMaterialCode()));
        if (exists) {
            throw new IllegalArgumentException("Material already exists: " + item.getMaterialCode());
        }
        this.items.add(item);
        this.updatedAt = LocalDateTime.now();
    }
    
    public void calculateFromBom(BomBillOfMaterials bom, Integer quantity) {
        if (bom == null || bom.getItems() == null) {
            throw new IllegalArgumentException("BOM is invalid");
        }
        
        this.productCode = bom.getProductCode();
        this.productName = bom.getProductName();
        this.quantity = quantity;
        this.bomVersion = bom.getVersion();
        
        // 清空现有项
        this.items.clear();
        
        // 计算每个物料的需求量
        for (BomItem item : bom.getActiveItems()) {
            Double requiredQty = item.calculateMaterialDemand(quantity);
            
            MaterialRequirementItem reqItem = new MaterialRequirementItem();
            reqItem.create(this.id, item.getMaterialCode(), item.getMaterialName(),
                          requiredQty, item.getUnitCode(), item.getUnitName());
            
            this.items.add(reqItem);
        }
        this.updatedAt = LocalDateTime.now();
    }
    
    public void confirm() {
        if (this.status.equals(RequirementStatus.DRAFT.name())) {
            this.status = RequirementStatus.CONFIRMED.name();
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public void cancel() {
        if (!this.status.equals(RequirementStatus.COMPLETED.name())) {
            this.status = RequirementStatus.DRAFT.name();
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public boolean isFullyIssued() {
        return items.stream()
                .allMatch(i -> MaterialRequirementItem.ItemStatus.ISSUED.name().equals(i.getStatus()));
    }
    
    public Double getIssueCompletionRate() {
        if (items.isEmpty()) {
            return 0.0;
        }
        double totalIssued = items.stream()
                .mapToDouble(MaterialRequirementItem::getIssuedQuantity)
                .sum();
        double totalRequired = items.stream()
                .mapToDouble(MaterialRequirementItem::getRequiredQuantity)
                .sum();
        if (totalRequired == 0) {
            return 0.0;
        }
        return (totalIssued / totalRequired) * 100;
    }
    
    public Map<String, Double> getPendingSummary() {
        Map<String, Double> summary = new HashMap<>();
        for (MaterialRequirementItem item : items) {
            summary.put(item.getMaterialCode(), item.getPendingQuantity());
        }
        return summary;
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRequirementNo() { return requirementNo; }
    public void setRequirementNo(String requirementNo) { this.requirementNo = requirementNo; }
    public String getWorkOrderNo() { return workOrderNo; }
    public void setWorkOrderNo(String workOrderNo) { this.workOrderNo = workOrderNo; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public Integer getQuantity() { return quantity; }
    public void setQuantity(Integer quantity) { this.quantity = quantity; }
    public String getBomVersion() { return bomVersion; }
    public void setBomVersion(String bomVersion) { this.bomVersion = bomVersion; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getWarehouseId() { return warehouseId; }
    public void setWarehouseId(String warehouseId) { this.warehouseId = warehouseId; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getRequirementType() { return requirementType; }
    public void setRequirementType(String requirementType) { this.requirementType = requirementType; }
    public LocalDateTime getRequiredDate() { return requiredDate; }
    public void setRequiredDate(LocalDateTime requiredDate) { this.requiredDate = requiredDate; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public List<MaterialRequirementItem> getItems() { return items; }
    public void setItems(List<MaterialRequirementItem> items) { this.items = items; }
}