package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class BomItem {
    
    private Long id;
    private Long bomId;
    private String materialCode;
    private String materialName;
    private String materialSpec;
    private String unitCode;
    private String unitName;
    private Double quantity;
    private Double scrapRate; // 报废率
    private Integer sequence;
    private String level; // 层级
    private String parentMaterialCode; // 父级物料编码（用于多层级BOM）
    private String itemType; // MATERIAL/COMPONENT/SUBASSEMBLY
    private String position; // 装配位置
    private String remark;
    private String status; // ACTIVE/INACTIVE
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    // 替代料关联（待关联）
    private Long substituteItemId;
    
    public enum ItemType {
        MATERIAL,      // 原材料
        COMPONENT,     // 组件
        SUBASSEMBLY    // 子总成
    }
    
    public enum ItemStatus {
        ACTIVE, INACTIVE
    }
    
    public void create(Long bomId, String materialCode, String materialName,
                      Double quantity, String unitCode, String unitName) {
        this.bomId = bomId;
        this.materialCode = materialCode;
        this.materialName = materialName;
        this.quantity = quantity;
        this.unitCode = unitCode;
        this.unitName = unitName;
        this.scrapRate = 0.0;
        this.itemType = ItemType.MATERIAL.name();
        this.status = ItemStatus.ACTIVE.name();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void createComponent(Long bomId, String materialCode, String materialName,
                                Double quantity, String unitCode, String unitName,
                                String parentMaterialCode) {
        create(bomId, materialCode, materialName, quantity, unitCode, unitName);
        this.itemType = ItemType.COMPONENT.name();
        this.parentMaterialCode = parentMaterialCode;
    }
    
    public void createSubassembly(Long bomId, String materialCode, String materialName,
                                  Double quantity, String unitCode, String unitName,
                                  String parentMaterialCode, Integer level) {
        create(bomId, materialCode, materialName, quantity, unitCode, unitName);
        this.itemType = ItemType.SUBASSEMBLY.name();
        this.parentMaterialCode = parentMaterialCode;
        this.level = level != null ? String.valueOf(level) : "1";
    }
    
    public void applyScrapRate(Double scrapRate) {
        if (scrapRate == null || scrapRate < 0 || scrapRate > 1) {
            throw new IllegalArgumentException("Scrap rate must be between 0 and 1");
        }
        this.scrapRate = scrapRate;
        this.updatedAt = LocalDateTime.now();
    }
    
    public Double calculateActualQuantity() {
        if (quantity == null || scrapRate == null) {
            return quantity;
        }
        // 实际用量 = 基本用量 / (1 - 报废率)
        return quantity / (1 - scrapRate);
    }
    
    public Double calculateMaterialDemand(Integer productQuantity) {
        Double actualQty = calculateActualQuantity();
        return actualQty * productQuantity.doubleValue();
    }
    
    public void setSubstituteItem(Long substituteItemId) {
        this.substituteItemId = substituteItemId;
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean hasSubstitute() {
        return substituteItemId != null;
    }
    
    public void deactivate() {
        this.status = ItemStatus.INACTIVE.name();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.status = ItemStatus.ACTIVE.name();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void updateQuantity(Double quantity) {
        if (quantity == null || quantity <= 0) {
            throw new IllegalArgumentException("Quantity must be positive");
        }
        this.quantity = quantity;
        this.updatedAt = LocalDateTime.now();
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getBomId() { return bomId; }
    public void setBomId(Long bomId) { this.bomId = bomId; }
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
    public Double getQuantity() { return quantity; }
    public void setQuantity(Double quantity) { this.quantity = quantity; }
    public Double getScrapRate() { return scrapRate; }
    public void setScrapRate(Double scrapRate) { this.scrapRate = scrapRate; }
    public Integer getSequence() { return sequence; }
    public void setSequence(Integer sequence) { this.sequence = sequence; }
    public String getLevel() { return level; }
    public void setLevel(String level) { this.level = level; }
    public String getParentMaterialCode() { return parentMaterialCode; }
    public void setParentMaterialCode(String parentMaterialCode) { this.parentMaterialCode = parentMaterialCode; }
    public String getItemType() { return itemType; }
    public void setItemType(String itemType) { this.itemType = itemType; }
    public String getPosition() { return position; }
    public void setPosition(String position) { this.position = position; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public Long getSubstituteItemId() { return substituteItemId; }
    public void setSubstituteItemId(Long substituteItemId) { this.substituteItemId = substituteItemId; }
}