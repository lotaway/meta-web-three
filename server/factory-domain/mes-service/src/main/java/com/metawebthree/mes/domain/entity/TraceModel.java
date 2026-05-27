package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class TraceModel {
    private Long id;
    private String modelCode;
    private String modelName;
    private String productType;
    private TraceRelationConfig relationConfig;
    private Boolean isEnabled;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public static class TraceRelationConfig {
        private Boolean enableBatchTrace;
        private Boolean enableSnTrace;
        private Boolean enableMaterialTrace;
        private Boolean enableProcessTrace;
        private Boolean enableQualityTrace;
        private Boolean enableEquipmentTrace;
        private List<TraceLevel> traceLevels;

        public static class TraceLevel {
            private String levelCode;
            private String levelName;
            private TraceType traceType;
            private String parentLevelCode;
            private Boolean isRequired;

            public String getLevelCode() { return levelCode; }
            public void setLevelCode(String levelCode) { this.levelCode = levelCode; }
            public String getLevelName() { return levelName; }
            public void setLevelName(String levelName) { this.levelName = levelName; }
            public TraceType getTraceType() { return traceType; }
            public void setTraceType(TraceType traceType) { this.traceType = traceType; }
            public String getParentLevelCode() { return parentLevelCode; }
            public void setParentLevelCode(String parentLevelCode) { this.parentLevelCode = parentLevelCode; }
            public Boolean getIsRequired() { return isRequired; }
            public void setIsRequired(Boolean isRequired) { this.isRequired = isRequired; }
        }

        public Boolean getEnableBatchTrace() { return enableBatchTrace; }
        public void setEnableBatchTrace(Boolean enableBatchTrace) { this.enableBatchTrace = enableBatchTrace; }
        public Boolean getEnableSnTrace() { return enableSnTrace; }
        public void setEnableSnTrace(Boolean enableSnTrace) { this.enableSnTrace = enableSnTrace; }
        public Boolean getEnableMaterialTrace() { return enableMaterialTrace; }
        public void setEnableMaterialTrace(Boolean enableMaterialTrace) { this.enableMaterialTrace = enableMaterialTrace; }
        public Boolean getEnableProcessTrace() { return enableProcessTrace; }
        public void setEnableProcessTrace(Boolean enableProcessTrace) { this.enableProcessTrace = enableProcessTrace; }
        public Boolean getEnableQualityTrace() { return enableQualityTrace; }
        public void setEnableQualityTrace(Boolean enableQualityTrace) { this.enableQualityTrace = enableQualityTrace; }
        public Boolean getEnableEquipmentTrace() { return enableEquipmentTrace; }
        public void setEnableEquipmentTrace(Boolean enableEquipmentTrace) { this.enableEquipmentTrace = enableEquipmentTrace; }
        public List<TraceLevel> getTraceLevels() { return traceLevels; }
        public void setTraceLevels(List<TraceLevel> traceLevels) { this.traceLevels = traceLevels; }
    }

    public enum TraceType {
        PRODUCT, BATCH, MATERIAL, SN, WORK_ORDER, PROCESS, QC, EQUIPMENT, OPERATOR
    }

    public void create(String modelCode, String modelName, String productType) {
        this.modelCode = modelCode;
        this.modelName = modelName;
        this.productType = productType;
        this.relationConfig = new TraceRelationConfig();
        this.relationConfig.setTraceLevels(new ArrayList<>());
        this.isEnabled = Boolean.TRUE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void enableBatchTrace(Boolean enable) {
        initRelationConfig();
        this.relationConfig.setEnableBatchTrace(enable);
        this.updatedAt = LocalDateTime.now();
    }

    public void enableSnTrace(Boolean enable) {
        initRelationConfig();
        this.relationConfig.setEnableSnTrace(enable);
        this.updatedAt = LocalDateTime.now();
    }

    public void enableMaterialTrace(Boolean enable) {
        initRelationConfig();
        this.relationConfig.setEnableMaterialTrace(enable);
        this.updatedAt = LocalDateTime.now();
    }

    public void addTraceLevel(String levelCode, String levelName, TraceType traceType, 
                              String parentLevelCode, Boolean isRequired) {
        initRelationConfig();
        TraceRelationConfig.TraceLevel level = new TraceRelationConfig.TraceLevel();
        level.setLevelCode(levelCode);
        level.setLevelName(levelName);
        level.setTraceType(traceType);
        level.setParentLevelCode(parentLevelCode);
        level.setIsRequired(isRequired);
        this.relationConfig.getTraceLevels().add(level);
        this.updatedAt = LocalDateTime.now();
    }

    public Optional<TraceRelationConfig.TraceLevel> findLevel(String levelCode) {
        if (relationConfig == null || relationConfig.getTraceLevels() == null) {
            return Optional.empty();
        }
        return relationConfig.getTraceLevels().stream()
            .filter(l -> levelCode.equals(l.getLevelCode()))
            .findFirst();
    }

    public void removeTraceLevel(String levelCode) {
        if (relationConfig != null && relationConfig.getTraceLevels() != null) {
            relationConfig.getTraceLevels().removeIf(l -> levelCode.equals(l.getLevelCode()));
            this.updatedAt = LocalDateTime.now();
        }
    }

    public void disable() {
        this.isEnabled = Boolean.FALSE;
        this.updatedAt = LocalDateTime.now();
    }

    public void enable() {
        this.isEnabled = Boolean.TRUE;
        this.updatedAt = LocalDateTime.now();
    }

    private void initRelationConfig() {
        if (this.relationConfig == null) {
            this.relationConfig = new TraceRelationConfig();
            this.relationConfig.setTraceLevels(new ArrayList<>());
        }
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getModelCode() { return modelCode; }
    public void setModelCode(String modelCode) { this.modelCode = modelCode; }
    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }
    public String getProductType() { return productType; }
    public void setProductType(String productType) { this.productType = productType; }
    public TraceRelationConfig getRelationConfig() { return relationConfig; }
    public void setRelationConfig(TraceRelationConfig relationConfig) { this.relationConfig = relationConfig; }
    public Boolean getIsEnabled() { return isEnabled; }
    public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}