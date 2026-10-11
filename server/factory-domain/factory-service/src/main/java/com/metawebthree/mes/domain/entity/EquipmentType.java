package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class EquipmentType {
    private Long id;
    private String typeCode;
    private String typeName;
    private String description;
    private String category;
    private Long statusMachineId;
    private String icon;
    private Integer sortOrder;
    private List<EquipmentTypeAttribute> attributes;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(String typeCode, String typeName, String category) {
        this.typeCode = typeCode;
        this.typeName = typeName;
        this.category = category;
        this.isActive = true;
        this.sortOrder = 0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.isActive = true;
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }

    public void addAttribute(EquipmentTypeAttribute attr) {
        if (this.attributes == null) {
            this.attributes = new ArrayList<>();
        }
        this.attributes.add(attr);
        this.updatedAt = LocalDateTime.now();
    }

    public void removeAttribute(String attrCode) {
        if (this.attributes == null) {
            return;
        }
        this.attributes.removeIf(a -> a.getAttributeCode().equals(attrCode));
        this.updatedAt = LocalDateTime.now();
    }

    public void bindStatusMachine(Long machineId) {
        this.statusMachineId = machineId;
        this.updatedAt = LocalDateTime.now();
    }

    public static class EquipmentTypeAttribute {
        private Long id;
        private Long equipmentTypeId;
        private String attributeCode;
        private String attributeName;
        private String dataType;
        private String defaultValue;
        private Boolean isRequired;
        private Boolean isUnique;
        private String validationRule;
        private String description;
        private Integer sortOrder;

        public void create(String attributeCode, String attributeName, String dataType) {
            this.attributeCode = attributeCode;
            this.attributeName = attributeName;
            this.dataType = dataType;
            this.isRequired = false;
            this.isUnique = false;
            this.sortOrder = 0;
        }

        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public Long getEquipmentTypeId() { return equipmentTypeId; }
        public void setEquipmentTypeId(Long equipmentTypeId) { this.equipmentTypeId = equipmentTypeId; }
        public String getAttributeCode() { return attributeCode; }
        public void setAttributeCode(String attributeCode) { this.attributeCode = attributeCode; }
        public String getAttributeName() { return attributeName; }
        public void setAttributeName(String attributeName) { this.attributeName = attributeName; }
        public String getDataType() { return dataType; }
        public void setDataType(String dataType) { this.dataType = dataType; }
        public String getDefaultValue() { return defaultValue; }
        public void setDefaultValue(String defaultValue) { this.defaultValue = defaultValue; }
        public Boolean getIsRequired() { return isRequired; }
        public void setIsRequired(Boolean isRequired) { this.isRequired = isRequired; }
        public Boolean getIsUnique() { return isUnique; }
        public void setIsUnique(Boolean isUnique) { this.isUnique = isUnique; }
        public String getValidationRule() { return validationRule; }
        public void setValidationRule(String validationRule) { this.validationRule = validationRule; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getSortOrder() { return sortOrder; }
        public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTypeCode() { return typeCode; }
    public void setTypeCode(String typeCode) { this.typeCode = typeCode; }
    public String getTypeName() { return typeName; }
    public void setTypeName(String typeName) { this.typeName = typeName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    public Long getStatusMachineId() { return statusMachineId; }
    public void setStatusMachineId(Long statusMachineId) { this.statusMachineId = statusMachineId; }
    public String getIcon() { return icon; }
    public void setIcon(String icon) { this.icon = icon; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public List<EquipmentTypeAttribute> getAttributes() { return attributes; }
    public void setAttributes(List<EquipmentTypeAttribute> attributes) { this.attributes = attributes; }
    public Boolean getIsActive() { return isActive; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}