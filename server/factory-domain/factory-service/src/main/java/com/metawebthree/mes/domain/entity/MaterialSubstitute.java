package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

public class MaterialSubstitute {
    
    private Long id;
    private String productCode;
    private String mainMaterialCode;
    private String mainMaterialName;
    private String status;
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private List<SubstituteItem> substitutes = new ArrayList<>();
    
    public enum SubstituteStatus {
        ACTIVE, INACTIVE
    }
    
    public static class SubstituteItem {
        private Long id;
        private Long substituteGroupId;
        private String materialCode;
        private String materialName;
        private String materialSpec;
        private String unitCode;
        private String unitName;
        private Integer priority;
        private Double conversionRate;
        private String conversionUnit;
        private String reason;
        private LocalDateTime effectiveDate;
        private LocalDateTime expiryDate;
        private String status;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;
        
        public enum ItemStatus {
            PENDING,
            ACTIVE,
            INACTIVE
        }
        
        public void create(Long substituteGroupId, String materialCode, String materialName,
                          Integer priority) {
            this.substituteGroupId = substituteGroupId;
            this.materialCode = materialCode;
            this.materialName = materialName;
            this.priority = priority;
            this.status = ItemStatus.PENDING.name();
            this.createdAt = LocalDateTime.now();
            this.updatedAt = LocalDateTime.now();
        }
        
        public void setConversionRate(Double conversionRate, String conversionUnit) {
            if (conversionRate == null || conversionRate <= 0) {
                throw new IllegalArgumentException("Conversion rate must be positive");
            }
            this.conversionRate = conversionRate;
            this.conversionUnit = conversionUnit;
            this.updatedAt = LocalDateTime.now();
        }
        
        public void activate(LocalDateTime effectiveDate) {
            this.effectiveDate = effectiveDate;
            this.status = ItemStatus.ACTIVE.name();
            this.updatedAt = LocalDateTime.now();
        }
        
        public void deactivate(String reason) {
            this.status = ItemStatus.INACTIVE.name();
            this.reason = reason;
            this.updatedAt = LocalDateTime.now();
        }
        
        public boolean isEffective(LocalDateTime date) {
            if (!ItemStatus.ACTIVE.name().equals(status)) {
                return false;
            }
            boolean afterEffective = effectiveDate == null || !date.isBefore(effectiveDate);
            boolean beforeExpiry = expiryDate == null || date.isBefore(expiryDate);
            return afterEffective && beforeExpiry;
        }
        
        public Double calculateSubstituteQuantity(Double mainMaterialQuantity) {
            if (conversionRate == null) {
                return mainMaterialQuantity;
            }
            return mainMaterialQuantity * conversionRate;
        }
        
        // Getters and Setters
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public Long getSubstituteGroupId() { return substituteGroupId; }
        public void setSubstituteGroupId(Long substituteGroupId) { this.substituteGroupId = substituteGroupId; }
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
        public Integer getPriority() { return priority; }
        public void setPriority(Integer priority) { this.priority = priority; }
        public Double getConversionRate() { return conversionRate; }
        public void setConversionRate(Double conversionRate) { this.conversionRate = conversionRate; }
        public String getConversionUnit() { return conversionUnit; }
        public void setConversionUnit(String conversionUnit) { this.conversionUnit = conversionUnit; }
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
        public LocalDateTime getEffectiveDate() { return effectiveDate; }
        public void setEffectiveDate(LocalDateTime effectiveDate) { this.effectiveDate = effectiveDate; }
        public LocalDateTime getExpiryDate() { return expiryDate; }
        public void setExpiryDate(LocalDateTime expiryDate) { this.expiryDate = expiryDate; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreatedAt() { return createdAt; }
        public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
        public LocalDateTime getUpdatedAt() { return updatedAt; }
        public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    }
    
    public void create(String productCode, String mainMaterialCode, String mainMaterialName) {
        this.productCode = productCode;
        this.mainMaterialCode = mainMaterialCode;
        this.mainMaterialName = mainMaterialName;
        this.status = SubstituteStatus.ACTIVE.name();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void addSubstitute(SubstituteItem item) {
        // 检查优先级是否重复
        boolean priorityExists = substitutes.stream()
                .anyMatch(s -> s.getPriority().equals(item.getPriority()));
        if (priorityExists) {
            throw new IllegalArgumentException("Priority already exists: " + item.getPriority());
        }
        this.substitutes.add(item);
        this.updatedAt = LocalDateTime.now();
    }
    
    public void removeSubstitute(Long substituteId) {
        this.substitutes.removeIf(s -> s.getId().equals(substituteId));
        this.updatedAt = LocalDateTime.now();
    }
    
    public Optional<SubstituteItem> getPrimarySubstitute(LocalDateTime date) {
        return this.substitutes.stream()
                .filter(s -> s.isEffective(date))
                .min(Comparator.comparing(SubstituteItem::getPriority));
    }
    
    public List<SubstituteItem> getActiveSubstitutes(LocalDateTime date) {
        return this.substitutes.stream()
                .filter(s -> s.isEffective(date))
                .sorted(Comparator.comparing(SubstituteItem::getPriority))
                .toList();
    }
    
    public int getSubstituteCount() {
        return this.substitutes.size();
    }
    
    public boolean isValidSubstitute(String materialCode, LocalDateTime date) {
        return this.substitutes.stream()
                .anyMatch(s -> s.getMaterialCode().equals(materialCode) && s.isEffective(date));
    }
    
    public void deactivate() {
        this.status = SubstituteStatus.INACTIVE.name();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        this.status = SubstituteStatus.ACTIVE.name();
        this.updatedAt = LocalDateTime.now();
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getMainMaterialCode() { return mainMaterialCode; }
    public void setMainMaterialCode(String mainMaterialCode) { this.mainMaterialCode = mainMaterialCode; }
    public String getMainMaterialName() { return mainMaterialName; }
    public void setMainMaterialName(String mainMaterialName) { this.mainMaterialName = mainMaterialName; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public List<SubstituteItem> getSubstitutes() { return substitutes; }
    public void setSubstitutes(List<SubstituteItem> substitutes) { this.substitutes = substitutes; }
    
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}