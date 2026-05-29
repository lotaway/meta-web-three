package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 产品类型实体
 * 支持产品分类定义、关联默认工艺路线、质检方案，以及属性继承
 */
public class ProductType {
    
    public enum TypeStatus {
        DRAFT,      // 草稿
        ACTIVE,     // 生效
        INACTIVE,   // 失效
        ARCHIVED    // 归档
    }
    
    /**
     * 继承的字段枚举 - 定义哪些字段可以被继承
     */
    public enum InheritableField {
        DEFAULT_PROCESS_ROUTE("defaultProcessRouteId", "默认工艺路线"),
        DEFAULT_QC_PLAN("defaultQcPlanId", "默认质检方案"),
        DEFAULT_PARAMETER_GROUP("defaultParameterGroupId", "默认参数组"),
        DEFAULT_ANDON_CONFIG("defaultAndonConfigId", "默认安灯配置"),
        DEFAULT_POKAYOKE_RULES("defaultPokayokeRuleIds", "默认防错规则"),
        DEFAULT_WORKSTATION_TYPE("defaultWorkstationType", "默认工位类型"),
        DEFAULT_INSPECTION_LEVEL("defaultInspectionLevel", "默认检验水平"),
        DEFAULT_SAMPLING_TYPE("defaultSamplingType", "默认抽样类型"),
        DEFAULT_AQL("defaultAql", "默认AQL"),
        TRACE_TEMPLATE("traceTemplateId", "追溯模板"),
        EQUIPMENT_REQUIREMENTS("equipmentRequirements", "设备要求"),
        MATERIAL_SPECS("materialSpecs", "物料规格");
        
        private final String fieldName;
        private final String description;
        
        InheritableField(String fieldName, String description) {
            this.fieldName = fieldName;
            this.description = description;
        }
        
        public String getFieldName() { return fieldName; }
        public String getDescription() { return description; }
    }
    
    private Long id;
    private String typeCode;                    // 产品类型编码
    private String typeName;                    // 产品类型名称
    private Long parentId;                      // 父类型ID（用于层级继承，null表示顶级类型）
    private String parentTypeCode;              // 父类型编码（冗余存储便于查询）
    private String category;                    // 产品大类（电子产品、机械产品、化工产品等）
    private String description;                 // 描述
    
    // 默认关联配置
    private Long defaultProcessRouteId;         // 默认工艺路线ID
    private String defaultProcessRouteCode;     // 默认工艺路线编码
    private Long defaultQcPlanId;               // 默认质检方案ID
    private String defaultQcPlanCode;           // 默认质检方案编码
    private Long defaultParameterGroupId;       // 默认参数组模板ID
    private String defaultParameterGroupCode;   // 默认参数组模板编码
    private Long defaultAndonConfigId;          // 默认安灯配置ID
    
    // 继承的字段配置（JSON格式存储）
    private Map<String, Object> inheritedFields;    // 从父类型继承的字段值
    private List<String> overriddenFields;          // 本类型覆盖的字段（不再继承）
    
    // 防错规则
    private List<Long> defaultPokayokeRuleIds;  // 默认防错规则ID列表
    private List<String> defaultPokayokeRuleCodes;
    
    // 追溯配置
    private Long traceTemplateId;               // 追溯模板ID
    private String traceTemplateCode;           // 追溯模板编码
    
    // 状态与配置
    private TypeStatus status;                  // 状态
    private Integer sortOrder;                  // 显示顺序
    private Boolean allowSubTypes;              // 是否允许有子类型
    private Integer maxSubTypeLevel;            // 最大子类型层级深度
    
    // 设备与物料要求
    private String defaultWorkstationType;      // 默认工位类型
    private String equipmentRequirements;       // 设备要求（JSON格式）
    private String materialSpecs;               // 物料规格要求（JSON格式）
    
    // 质检默认配置
    private String defaultInspectionLevel;      // 默认检验水平
    private String defaultSamplingType;         // 默认抽样类型
    private String defaultAql;                  // 默认AQL
    
    private String remark;                      // 备注
    
    private String createdBy;
    private LocalDateTime createdAt;
    private String updatedBy;
    private LocalDateTime updatedAt;
    
    // 运行时属性（非持久化）
    private List<ProductType> subTypes = new ArrayList<>();      // 子类型列表
    private ProductType parentType;                               // 父类型对象
    private Map<InheritableField, Object> effectiveConfig;       // 生效的配置（包含继承值）
    
    /**
     * 创建产品类型
     */
    public void create(String typeCode, String typeName, String category) {
        this.typeCode = typeCode;
        this.typeName = typeName;
        this.category = category;
        this.status = TypeStatus.DRAFT;
        this.allowSubTypes = true;
        this.maxSubTypeLevel = 3;
        this.inheritedFields = new java.util.HashMap<>();
        this.overriddenFields = new ArrayList<>();
        this.defaultPokayokeRuleIds = new ArrayList<>();
        this.defaultPokayokeRuleCodes = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 激活产品类型
     */
    public void activate() {
        if (this.status == TypeStatus.DRAFT || this.status == TypeStatus.INACTIVE) {
            this.status = TypeStatus.ACTIVE;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    /**
     * 停用产品类型
     */
    public void deactivate() {
        if (this.status == TypeStatus.ACTIVE) {
            this.status = TypeStatus.INACTIVE;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    /**
     * 归档产品类型
     */
    public void archive() {
        if (this.status != TypeStatus.ARCHIVED) {
            this.status = TypeStatus.ARCHIVED;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    /**
     * 设置父类型（建立继承关系）
     */
    public void setParent(ProductType parent) {
        if (parent == null) {
            this.parentId = null;
            this.parentTypeCode = null;
            this.parentType = null;
            this.inheritedFields.clear();
            return;
        }
        
        // 检查循环引用
        if (wouldCreateCycle(parent)) {
            throw new IllegalStateException("设置父类型会创建循环引用");
        }
        
        this.parentId = parent.getId();
        this.parentTypeCode = parent.getTypeCode();
        this.parentType = parent;
        
        // 继承父类型的有效配置
        inheritFromParent(parent);
    }
    
    /**
     * 检查是否会创建循环引用
     */
    private boolean wouldCreateCycle(ProductType potentialParent) {
        ProductType current = potentialParent;
        while (current != null) {
            if (current.getId().equals(this.id)) {
                return true;
            }
            current = current.getParentType();
        }
        return false;
    }
    
    /**
     * 从父类型继承配置
     */
    private void inheritFromParent(ProductType parent) {
        Map<InheritableField, Object> parentConfig = parent.getEffectiveConfig();
        
        // 复制父类型的继承配置
        for (Map.Entry<InheritableField, Object> entry : parentConfig.entrySet()) {
            InheritableField field = entry.getKey();
            // 如果本类型没有覆盖该字段，则继承
            if (!this.overriddenFields.contains(field.getFieldName())) {
                this.inheritedFields.put(field.getFieldName(), entry.getValue());
            }
        }
    }
    
    /**
     * 覆盖某个字段（不再继承父类型）
     */
    public void overrideField(InheritableField field, Object value) {
        this.inheritedFields.put(field.getFieldName(), value);
        if (!this.overriddenFields.contains(field.getFieldName())) {
            this.overriddenFields.add(field.getFieldName());
        }
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 恢复继承（取消覆盖）
     */
    public void restoreInheritance(InheritableField field) {
        this.overriddenFields.remove(field.getFieldName());
        this.inheritedFields.remove(field.getFieldName());
        
        // 重新从父类型继承
        if (this.parentType != null) {
            Map<InheritableField, Object> parentConfig = this.parentType.getEffectiveConfig();
            Object parentValue = parentConfig.get(field);
            if (parentValue != null) {
                this.inheritedFields.put(field.getFieldName(), parentValue);
            }
        }
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 获取生效的配置（包含继承值）
     * 优先级：本类型值 > 父类型继承值
     */
    public Map<InheritableField, Object> getEffectiveConfig() {
        Map<InheritableField, Object> config = new java.util.HashMap<>();
        
        // 首先收集所有父类型的有效配置
        if (parentType != null) {
            config.putAll(parentType.getEffectiveConfig());
        } else if (!this.inheritedFields.isEmpty()) {
            // 顶级类型，从自身继承字段初始化
            for (Map.Entry<String, Object> entry : this.inheritedFields.entrySet()) {
                try {
                    InheritableField field = InheritableField.valueOf(entry.getKey());
                    config.put(field, entry.getValue());
                } catch (IllegalArgumentException e) {
                    // 忽略未知字段
                }
            }
        }
        
        // 然后用本类型的值覆盖
        for (InheritableField field : InheritableField.values()) {
            Object localValue = getFieldValue(field);
            if (localValue != null) {
                config.put(field, localValue);
            }
        }
        
        return config;
    }
    
    /**
     * 获取指定字段的值（优先本类型，再继承）
     */
    private Object getFieldValue(InheritableField field) {
        return getLocalFieldValue(field);
    }
    
    /**
     * 获取指定字段的值（优先本类型，再继承）
     */
    private Object getLocalFieldValue(InheritableField field) {
        // 首先检查本类型的直接值
        Object directValue = getDirectFieldValue(field);
        if (directValue != null) {
            return directValue;
        }
        
        // 然后检查继承的值
        Object inheritedValue = this.inheritedFields.get(field.getFieldName());
        if (inheritedValue != null) {
            return inheritedValue;
        }
        
        // 最后从父类型递归获取
        if (parentType != null) {
            return parentType.getLocalFieldValue(field);
        }
        
        return null;
    }
    
    /**
     * 获取本类型的直接字段值
     */
    private Object getDirectFieldValue(InheritableField field) {
        switch (field) {
            case DEFAULT_PROCESS_ROUTE:
                return this.defaultProcessRouteId;
            case DEFAULT_QC_PLAN:
                return this.defaultQcPlanId;
            case DEFAULT_PARAMETER_GROUP:
                return this.defaultParameterGroupId;
            case DEFAULT_ANDON_CONFIG:
                return this.defaultAndonConfigId;
            case DEFAULT_POKAYOKE_RULES:
                return this.defaultPokayokeRuleIds;
            case DEFAULT_WORKSTATION_TYPE:
                return this.defaultWorkstationType;
            case DEFAULT_INSPECTION_LEVEL:
                return this.defaultInspectionLevel;
            case DEFAULT_SAMPLING_TYPE:
                return this.defaultSamplingType;
            case DEFAULT_AQL:
                return this.defaultAql;
            case TRACE_TEMPLATE:
                return this.traceTemplateId;
            case EQUIPMENT_REQUIREMENTS:
                return this.equipmentRequirements;
            case MATERIAL_SPECS:
                return this.materialSpecs;
            default:
                return null;
        }
    }
    
    /**
     * 获取生效的工艺路线ID
     */
    public Long getEffectiveProcessRouteId() {
        Object value = getEffectiveConfig().get(InheritableField.DEFAULT_PROCESS_ROUTE);
        return value instanceof Long ? (Long) value : null;
    }
    
    /**
     * 获取生效的质检方案ID
     */
    public Long getEffectiveQcPlanId() {
        Object value = getEffectiveConfig().get(InheritableField.DEFAULT_QC_PLAN);
        return value instanceof Long ? (Long) value : null;
    }
    
    /**
     * 获取生效的参数组ID
     */
    public Long getEffectiveParameterGroupId() {
        Object value = getEffectiveConfig().get(InheritableField.DEFAULT_PARAMETER_GROUP);
        return value instanceof Long ? (Long) value : null;
    }
    
    /**
     * 获取生效的安灯配置ID
     */
    public Long getEffectiveAndonConfigId() {
        Object value = getEffectiveConfig().get(InheritableField.DEFAULT_ANDON_CONFIG);
        return value instanceof Long ? (Long) value : null;
    }
    
    /**
     * 检查是否为顶级类型
     */
    public boolean isTopLevel() {
        return this.parentId == null;
    }
    
    /**
     * 检查是否有子类型
     */
    public boolean hasSubTypes() {
        return this.subTypes != null && !this.subTypes.isEmpty();
    }
    
    /**
     * 添加子类型
     */
    public void addSubType(ProductType subType) {
        if (this.subTypes == null) {
            this.subTypes = new ArrayList<>();
        }
        if (!this.subTypes.contains(subType)) {
            this.subTypes.add(subType);
            subType.setParent(this);
        }
    }
    
    /**
     * 移除子类型
     */
    public void removeSubType(ProductType subType) {
        if (this.subTypes != null) {
            this.subTypes.remove(subType);
        }
    }
    
    /**
     * 获取继承路径（从顶级到当前）
     */
    public List<ProductType> getInheritancePath() {
        List<ProductType> path = new ArrayList<>();
        ProductType current = this;
        while (current != null) {
            path.add(0, current);
            current = current.getParentType();
        }
        return path;
    }
    
    /**
     * 验证类型是否有效
     */
    public boolean isValid() {
        return typeCode != null && !typeCode.isEmpty()
            && typeName != null && !typeName.isEmpty()
            && category != null && !category.isEmpty();
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public String getTypeCode() { return typeCode; }
    public void setTypeCode(String typeCode) { this.typeCode = typeCode; }
    
    public String getTypeName() { return typeName; }
    public void setTypeName(String typeName) { this.typeName = typeName; }
    
    public Long getParentId() { return parentId; }
    public void setParentId(Long parentId) { this.parentId = parentId; }
    
    public String getParentTypeCode() { return parentTypeCode; }
    public void setParentTypeCode(String parentTypeCode) { this.parentTypeCode = parentTypeCode; }
    
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    
    public Long getDefaultProcessRouteId() { return defaultProcessRouteId; }
    public void setDefaultProcessRouteId(Long defaultProcessRouteId) { this.defaultProcessRouteId = defaultProcessRouteId; }
    
    public String getDefaultProcessRouteCode() { return defaultProcessRouteCode; }
    public void setDefaultProcessRouteCode(String defaultProcessRouteCode) { this.defaultProcessRouteCode = defaultProcessRouteCode; }
    
    public Long getDefaultQcPlanId() { return defaultQcPlanId; }
    public void setDefaultQcPlanId(Long defaultQcPlanId) { this.defaultQcPlanId = defaultQcPlanId; }
    
    public String getDefaultQcPlanCode() { return defaultQcPlanCode; }
    public void setDefaultQcPlanCode(String defaultQcPlanCode) { this.defaultQcPlanCode = defaultQcPlanCode; }
    
    public Long getDefaultParameterGroupId() { return defaultParameterGroupId; }
    public void setDefaultParameterGroupId(Long defaultParameterGroupId) { this.defaultParameterGroupId = defaultParameterGroupId; }
    
    public String getDefaultParameterGroupCode() { return defaultParameterGroupCode; }
    public void setDefaultParameterGroupCode(String defaultParameterGroupCode) { this.defaultParameterGroupCode = defaultParameterGroupCode; }
    
    public Long getDefaultAndonConfigId() { return defaultAndonConfigId; }
    public void setDefaultAndonConfigId(Long defaultAndonConfigId) { this.defaultAndonConfigId = defaultAndonConfigId; }
    
    public Map<String, Object> getInheritedFields() { return inheritedFields; }
    public void setInheritedFields(Map<String, Object> inheritedFields) { this.inheritedFields = inheritedFields; }
    
    public List<String> getOverriddenFields() { return overriddenFields; }
    public void setOverriddenFields(List<String> overriddenFields) { this.overriddenFields = overriddenFields; }
    
    public List<Long> getDefaultPokayokeRuleIds() { return defaultPokayokeRuleIds; }
    public void setDefaultPokayokeRuleIds(List<Long> defaultPokayokeRuleIds) { this.defaultPokayokeRuleIds = defaultPokayokeRuleIds; }
    
    public List<String> getDefaultPokayokeRuleCodes() { return defaultPokayokeRuleCodes; }
    public void setDefaultPokayokeRuleCodes(List<String> defaultPokayokeRuleCodes) { this.defaultPokayokeRuleCodes = defaultPokayokeRuleCodes; }
    
    public Long getTraceTemplateId() { return traceTemplateId; }
    public void setTraceTemplateId(Long traceTemplateId) { this.traceTemplateId = traceTemplateId; }
    
    public String getTraceTemplateCode() { return traceTemplateCode; }
    public void setTraceTemplateCode(String traceTemplateCode) { this.traceTemplateCode = traceTemplateCode; }
    
    public TypeStatus getStatus() { return status; }
    public void setStatus(TypeStatus status) { this.status = status; }
    
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    
    public Boolean getAllowSubTypes() { return allowSubTypes; }
    public void setAllowSubTypes(Boolean allowSubTypes) { this.allowSubTypes = allowSubTypes; }
    
    public Integer getMaxSubTypeLevel() { return maxSubTypeLevel; }
    public void setMaxSubTypeLevel(Integer maxSubTypeLevel) { this.maxSubTypeLevel = maxSubTypeLevel; }
    
    public String getDefaultWorkstationType() { return defaultWorkstationType; }
    public void setDefaultWorkstationType(String defaultWorkstationType) { this.defaultWorkstationType = defaultWorkstationType; }
    
    public String getEquipmentRequirements() { return equipmentRequirements; }
    public void setEquipmentRequirements(String equipmentRequirements) { this.equipmentRequirements = equipmentRequirements; }
    
    public String getMaterialSpecs() { return materialSpecs; }
    public void setMaterialSpecs(String materialSpecs) { this.materialSpecs = materialSpecs; }
    
    public String getDefaultInspectionLevel() { return defaultInspectionLevel; }
    public void setDefaultInspectionLevel(String defaultInspectionLevel) { this.defaultInspectionLevel = defaultInspectionLevel; }
    
    public String getDefaultSamplingType() { return defaultSamplingType; }
    public void setDefaultSamplingType(String defaultSamplingType) { this.defaultSamplingType = defaultSamplingType; }
    
    public String getDefaultAql() { return defaultAql; }
    public void setDefaultAql(String defaultAql) { this.defaultAql = defaultAql; }
    
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
    
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    
    public List<ProductType> getSubTypes() { return subTypes; }
    public void setSubTypes(List<ProductType> subTypes) { this.subTypes = subTypes; }
    
    public ProductType getParentType() { return parentType; }
    public void setParentType(ProductType parentType) { this.parentType = parentType; }
}