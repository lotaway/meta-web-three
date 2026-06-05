package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import com.metawebthree.mes.domain.QcConstants;

public class TraceDataScope {
    private Long id;
    private String scopeCode;
    private String scopeName;
    private DataScopeType scopeType;
    private List<ScopeItem> items;
    private Boolean isDefault;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum DataScopeType {
        PROCESS_PARAMETER, QC_RESULT, OPERATOR, EQUIPMENT, ENVIRONMENT, MATERIAL_CONSUMPTION
    }

    public static class ScopeItem {
        private String itemCode;
        private String itemName;
        private DataScopeType dataType;
        private Boolean isRequired;
        private Integer retentionDays;
        private String description;

        public String getItemCode() { return itemCode; }
        public void setItemCode(String itemCode) { this.itemCode = itemCode; }
        public String getItemName() { return itemName; }
        public void setItemName(String itemName) { this.itemName = itemName; }
        public DataScopeType getDataType() { return dataType; }
        public void setDataType(DataScopeType dataType) { this.dataType = dataType; }
        public Boolean getIsRequired() { return isRequired; }
        public void setIsRequired(Boolean isRequired) { this.isRequired = isRequired; }
        public Integer getRetentionDays() { return retentionDays; }
        public void setRetentionDays(Integer retentionDays) { this.retentionDays = retentionDays; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }

    public void create(String scopeCode, String scopeName, DataScopeType scopeType) {
        this.scopeCode = scopeCode;
        this.scopeName = scopeName;
        this.scopeType = scopeType;
        this.items = new ArrayList<>();
        this.isDefault = Boolean.FALSE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void addItem(String itemCode, String itemName, DataScopeType dataType, 
                        Boolean isRequired, Integer retentionDays) {
        if (this.items == null) {
            this.items = new ArrayList<>();
        }
        ScopeItem item = new ScopeItem();
        item.setItemCode(itemCode);
        item.setItemName(itemName);
        item.setDataType(dataType);
        item.setIsRequired(isRequired);
        item.setRetentionDays(retentionDays);
        this.items.add(item);
        this.updatedAt = LocalDateTime.now();
    }

    public void addProcessParameter(String itemCode, String itemName, Boolean isRequired) {
        addItem(itemCode, itemName, DataScopeType.PROCESS_PARAMETER, isRequired, QcConstants.RETENTION_DAYS_PROCESS);
    }

    public void addQcResult(String itemCode, String itemName, Boolean isRequired) {
        addItem(itemCode, itemName, DataScopeType.QC_RESULT, isRequired, QcConstants.RETENTION_DAYS_QC_RESULT);
    }

    public void addOperator(String itemCode, String itemName, Boolean isRequired) {
        addItem(itemCode, itemName, DataScopeType.OPERATOR, isRequired, QcConstants.RETENTION_DAYS_OPERATOR);
    }

    public void addEquipment(String itemCode, String itemName, Boolean isRequired) {
        addItem(itemCode, itemName, DataScopeType.EQUIPMENT, isRequired, QcConstants.RETENTION_DAYS_EQUIPMENT);
    }

    public Optional<ScopeItem> findItem(String itemCode) {
        if (items == null) {
            return Optional.empty();
        }
        return items.stream()
            .filter(i -> itemCode.equals(i.getItemCode()))
            .findFirst();
    }

    public void removeItem(String itemCode) {
        if (items != null) {
            items.removeIf(i -> itemCode.equals(i.getItemCode()));
            this.updatedAt = LocalDateTime.now();
        }
    }

    public void setAsDefault() {
        this.isDefault = Boolean.TRUE;
        this.updatedAt = LocalDateTime.now();
    }

    public void unsetAsDefault() {
        this.isDefault = Boolean.FALSE;
        this.updatedAt = LocalDateTime.now();
    }

    public List<ScopeItem> getRequiredItems() {
        if (items == null) {
            return new ArrayList<>();
        }
        return items.stream()
            .filter(i -> Boolean.TRUE.equals(i.getIsRequired()))
            .toList();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getScopeCode() { return scopeCode; }
    public void setScopeCode(String scopeCode) { this.scopeCode = scopeCode; }
    public String getScopeName() { return scopeName; }
    public void setScopeName(String scopeName) { this.scopeName = scopeName; }
    public DataScopeType getScopeType() { return scopeType; }
    public void setScopeType(DataScopeType scopeType) { this.scopeType = scopeType; }
    public List<ScopeItem> getItems() { return items; }
    public void setItems(List<ScopeItem> items) { this.items = items; }
    public Boolean getIsDefault() { return isDefault; }
    public void setIsDefault(Boolean isDefault) { this.isDefault = isDefault; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}