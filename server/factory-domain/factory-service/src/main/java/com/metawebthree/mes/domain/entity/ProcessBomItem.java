package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class ProcessBomItem {
    
    private Long id;
    private String processBomCode;
    private String productCode;
    private String processRouteId;
    private String processCode; // 工序编码
    private String processName; // 工序名称
    private String version;
    private String status; // ACTIVE/INACTIVE
    private String description;
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    // 工序级物料列表
    private List<ProcessMaterial> materials = new ArrayList<>();
    
    public enum ProcessBomStatus {
        ACTIVE, INACTIVE
    }
    
    public static class ProcessMaterial {
        private Long id;
        private Long processBomId;
        private String materialCode;
        private String materialName;
        private String materialSpec;
        private String unitCode;
        private String unitName;
        private Double quantity;
        private Double scrapRate;
        private Integer sequence;
        private String materialType; // CONSUMABLE/MAIN/AUXILIARY
        private String status;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;
        
        public enum MaterialType {
            CONSUMABLE,  // 消耗品
            MAIN,       // 主材
            AUXILIARY   // 辅材
        }
        
        public void create(Long processBomId, String materialCode, String materialName,
                          Double quantity, String unitCode, String unitName) {
            this.processBomId = processBomId;
            this.materialCode = materialCode;
            this.materialName = materialName;
            this.quantity = quantity;
            this.unitCode = unitCode;
            this.unitName = unitName;
            this.scrapRate = 0.0;
            this.materialType = MaterialType.MAIN.name();
            this.status = "ACTIVE";
            this.createdAt = LocalDateTime.now();
            this.updatedAt = LocalDateTime.now();
        }
        
        public Double calculateActualQuantity() {
            if (quantity == null || scrapRate == null) {
                return quantity;
            }
            return quantity / (1 - scrapRate);
        }
        
        public Double calculateMaterialDemand(Integer productQuantity) {
            return calculateActualQuantity() * productQuantity.doubleValue();
        }
        
        // Getters and Setters
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public Long getProcessBomId() { return processBomId; }
        public void setProcessBomId(Long processBomId) { this.processBomId = processBomId; }
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
        public String getMaterialType() { return materialType; }
        public void setMaterialType(String materialType) { this.materialType = materialType; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreatedAt() { return createdAt; }
        public LocalDateTime getUpdatedAt() { return updatedAt; }
    }
    
    public void create(String processBomCode, String productCode, 
                      String processRouteId, String processCode, String processName) {
        this.processBomCode = processBomCode;
        this.productCode = productCode;
        this.processRouteId = processRouteId;
        this.processCode = processCode;
        this.processName = processName;
        this.version = "1.0";
        this.status = ProcessBomStatus.ACTIVE.name();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void addMaterial(ProcessMaterial material) {
        this.materials.add(material);
        this.updatedAt = LocalDateTime.now();
    }
    
    public Optional<ProcessMaterial> findMaterialByCode(String materialCode) {
        return this.materials.stream()
                .filter(m -> m.getMaterialCode().equals(materialCode))
                .findFirst();
    }
    
    public List<ProcessMaterial> getActiveMaterials() {
        return this.materials.stream()
                .filter(m -> "ACTIVE".equals(m.getStatus()))
                .toList();
    }
    
    public Double calculateMaterialDemand(String materialCode, Integer productQuantity) {
        return findMaterialByCode(materialCode)
                .map(m -> m.calculateMaterialDemand(productQuantity))
                .orElse(0.0);
    }
    
    public void deactivate() {
        this.status = ProcessBomStatus.INACTIVE.name();
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean validate() {
        if (processBomCode == null || productCode == null || processCode == null) {
            return false;
        }
        return !materials.isEmpty();
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getProcessBomCode() { return processBomCode; }
    public void setProcessBomCode(String processBomCode) { this.processBomCode = processBomCode; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProcessRouteId() { return processRouteId; }
    public void setProcessRouteId(String processRouteId) { this.processRouteId = processRouteId; }
    public String getProcessCode() { return processCode; }
    public void setProcessCode(String processCode) { this.processCode = processCode; }
    public String getProcessName() { return processName; }
    public void setProcessName(String processName) { this.processName = processName; }
    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public List<ProcessMaterial> getMaterials() { return materials; }
    public void setMaterials(List<ProcessMaterial> materials) { this.materials = materials; }
    
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}