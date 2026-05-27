package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class BomBillOfMaterials {
    
    private Long id;
    private String bomCode;
    private String productCode;
    private String productName;
    private String version;
    private String versionStatus;
    private LocalDateTime effectiveDate;
    private LocalDateTime expiryDate;
    private String bomType;
    private String processRouteId;
    private String description;
    private String status;
    private Integer itemCount;
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    // 关联的BOM子项
    private List<BomItem> items = new ArrayList<>();
    
    // 版本管理相关
    private String previousVersion;
    private String changeReason;
    
    public enum VersionStatus {
        DRAFT, ACTIVE, DEPRECATED
    }
    
    public enum BomType {
        MAIN,       // 顶层BOM
        PROCESS,    // 工序BOM
        SUBSTITUTE  // 替代料BOM
    }
    
    public enum BomStatus {
        ACTIVE, INACTIVE
    }
    
    public void create(String bomCode, String productCode, String productName, 
                      String version, String bomType) {
        this.bomCode = bomCode;
        this.productCode = productCode;
        this.productName = productName;
        this.version = version;
        this.bomType = bomType;
        this.versionStatus = VersionStatus.DRAFT.name();
        this.status = BomStatus.ACTIVE.name();
        this.itemCount = 0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        if (this.versionStatus.equals(VersionStatus.DRAFT.name())) {
            this.versionStatus = VersionStatus.ACTIVE.name();
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public void deprecate(String reason) {
        this.versionStatus = VersionStatus.DEPRECATED.name();
        this.changeReason = reason;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void applyEffectiveDate(LocalDateTime effectiveDate) {
        this.effectiveDate = effectiveDate;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void applyExpiryDate(LocalDateTime expiryDate) {
        this.expiryDate = expiryDate;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void addItem(BomItem item) {
        this.items.add(item);
        this.itemCount = this.items.size();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void removeItem(Long itemId) {
        this.items.removeIf(item -> item.getId().equals(itemId));
        this.itemCount = this.items.size();
        this.updatedAt = LocalDateTime.now();
    }
    
    public Optional<BomItem> findItemByMaterialCode(String materialCode) {
        return this.items.stream()
                .filter(item -> item.getMaterialCode().equals(materialCode))
                .findFirst();
    }
    
    public List<BomItem> getActiveItems() {
        return this.items.stream()
                .filter(item -> "ACTIVE".equals(item.getStatus()))
                .toList();
    }
    
    public boolean validate() {
        if (bomCode == null || productCode == null || version == null) {
            return false;
        }
        if (items.isEmpty()) {
            return false;
        }
        // 验证子项物料编码不重复
        long distinctCount = items.stream()
                .map(BomItem::getMaterialCode)
                .distinct()
                .count();
        return distinctCount == items.size();
    }
    
    public Double calculateTotalMaterialQuantity(String materialCode, Integer productQuantity) {
        return findItemByMaterialCode(materialCode)
                .map(item -> item.getQuantity() * productQuantity.doubleValue())
                .orElse(0.0);
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getBomCode() { return bomCode; }
    public void setBomCode(String bomCode) { this.bomCode = bomCode; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }
    public String getVersionStatus() { return versionStatus; }
    public void setVersionStatus(String versionStatus) { this.versionStatus = versionStatus; }
    public LocalDateTime getEffectiveDate() { return effectiveDate; }
    public void setEffectiveDate(LocalDateTime effectiveDate) { this.effectiveDate = effectiveDate; }
    public LocalDateTime getExpiryDate() { return expiryDate; }
    public void setExpiryDate(LocalDateTime expiryDate) { this.expiryDate = expiryDate; }
    public String getBomType() { return bomType; }
    public void setBomType(String bomType) { this.bomType = bomType; }
    public String getProcessRouteId() { return processRouteId; }
    public void setProcessRouteId(String processRouteId) { this.processRouteId = processRouteId; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public Integer getItemCount() { return itemCount; }
    public void setItemCount(Integer itemCount) { this.itemCount = itemCount; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public List<BomItem> getItems() { return items; }
    public void setItems(List<BomItem> items) { 
        this.items = items; 
        this.itemCount = items != null ? items.size() : 0;
    }
    public String getPreviousVersion() { return previousVersion; }
    public void setPreviousVersion(String previousVersion) { this.previousVersion = previousVersion; }
    public String getChangeReason() { return changeReason; }
    public void setChangeReason(String changeReason) { this.changeReason = changeReason; }
}