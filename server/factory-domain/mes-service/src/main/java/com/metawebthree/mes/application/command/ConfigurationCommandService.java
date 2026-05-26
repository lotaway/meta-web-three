package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.CodeRule;
import com.metawebthree.mes.domain.entity.DataDictionary;
import com.metawebthree.mes.domain.entity.EntityExtensionField;
import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import com.metawebthree.mes.domain.repository.CodeRuleRepository;
import com.metawebthree.mes.domain.repository.DataDictionaryRepository;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldRepository;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldValueRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 配置管理命令服务
 * 负责处理扩展字段、数据字典、编码规则的配置管理
 */
@Service
public class ConfigurationCommandService {
    
    private final EntityExtensionFieldRepository fieldRepository;
    private final EntityExtensionFieldValueRepository fieldValueRepository;
    private final DataDictionaryRepository dictionaryRepository;
    private final CodeRuleRepository codeRuleRepository;
    
    public ConfigurationCommandService(
            EntityExtensionFieldRepository fieldRepository,
            EntityExtensionFieldValueRepository fieldValueRepository,
            DataDictionaryRepository dictionaryRepository,
            CodeRuleRepository codeRuleRepository) {
        this.fieldRepository = fieldRepository;
        this.fieldValueRepository = fieldValueRepository;
        this.dictionaryRepository = dictionaryRepository;
        this.codeRuleRepository = codeRuleRepository;
    }
    
    // ==================== 扩展字段管理 ====================
    
    /**
     * 创建扩展字段定义
     */
    public Long createExtensionField(String entityType, String fieldCode, String fieldName,
                                     String fieldType, String fieldGroup, Boolean required,
                                     String defaultValue, String validationRule) {
        
        if (fieldRepository.existsByEntityTypeAndFieldCode(entityType, fieldCode)) {
            throw new IllegalArgumentException("Field code already exists: " + fieldCode);
        }
        
        EntityExtensionField field = new EntityExtensionField();
        EntityExtensionField.FieldType type = EntityExtensionField.FieldType.valueOf(fieldType);
        field.create(entityType, fieldCode, fieldName, type, fieldGroup);
        field.setRequired(required != null ? required : false);
        field.setDefaultValue(defaultValue);
        field.setValidationRule(validationRule);
        
        return fieldRepository.save(field).getId();
    }
    
    /**
     * 更新扩展字段定义
     */
    public void updateExtensionField(Long id, String fieldName, String fieldGroup,
                                     Boolean required, String defaultValue, String validationRule,
                                     Boolean listVisible, Boolean searchable) {
        
        EntityExtensionField field = fieldRepository.findById(id);
        if (field == null) {
            throw new IllegalArgumentException("Extension field not found: " + id);
        }
        
        if (fieldName != null) field.setFieldName(fieldName);
        if (fieldGroup != null) field.setFieldGroup(fieldGroup);
        if (required != null) field.setRequired(required);
        if (defaultValue != null) field.setDefaultValue(defaultValue);
        if (validationRule != null) field.setValidationRule(validationRule);
        if (listVisible != null) field.setListVisible(listVisible);
        if (searchable != null) field.setSearchable(searchable);
        
        fieldRepository.save(field);
    }
    
    /**
     * 删除扩展字段定义
     */
    public void deleteExtensionField(Long id) {
        fieldRepository.delete(id);
    }
    
    /**
     * 设置实体扩展字段值
     */
    public void setExtensionFieldValue(String entityType, Long entityId, 
                                       String fieldCode, String fieldValue) {
        // 验证字段是否存在且有效
        EntityExtensionField field = fieldRepository.findByEntityTypeAndFieldCode(entityType, fieldCode);
        if (field == null) {
            throw new IllegalArgumentException("Extension field not defined: " + fieldCode);
        }
        
        // 校验值
        if (!field.validateValue(fieldValue)) {
            throw new IllegalArgumentException("Field value validation failed: " + fieldCode);
        }
        
        EntityExtensionFieldValue value = new EntityExtensionFieldValue();
        value.create(entityType, entityId, fieldCode, fieldValue);
        fieldValueRepository.save(value);
    }
    
    /**
     * 批量设置实体扩展字段值
     */
    public void setExtensionFieldValues(String entityType, Long entityId,
                                        Map<String, String> fieldValues) {
        for (Map.Entry<String, String> entry : fieldValues.entrySet()) {
            setExtensionFieldValue(entityType, entityId, entry.getKey(), entry.getValue());
        }
    }
    
    // ==================== 数据字典管理 ====================
    
    /**
     * 创建数据字典
     */
    public Long createDataDictionary(String dictCode, String dictName, String description) {
        
        if (dictionaryRepository.existsByDictCode(dictCode)) {
            throw new IllegalArgumentException("Dictionary code already exists: " + dictCode);
        }
        
        DataDictionary dictionary = new DataDictionary();
        dictionary.create(dictCode, dictName, description);
        
        return dictionaryRepository.save(dictionary).getId();
    }
    
    /**
     * 添加字典项
     */
    public void addDictionaryItem(Long dictId, String itemCode, String itemLabel,
                                  String parentItemCode, Integer sortOrder) {
        
        DataDictionary dictionary = dictionaryRepository.findById(dictId)
                .orElseThrow(() -> new IllegalArgumentException("Dictionary not found: " + dictId));
        
        DataDictionary.DataDictionaryItem item = new DataDictionary.DataDictionaryItem();
        item.create(dictId, itemCode, itemLabel, sortOrder);
        item.setParentItemCode(parentItemCode);
        
        dictionary.getItems().add(item);
        dictionaryRepository.save(dictionary);
    }
    
    /**
     * 更新字典项
     */
    public void updateDictionaryItem(Long dictId, String itemCode, String itemLabel, Integer sortOrder) {
        
        DataDictionary dictionary = dictionaryRepository.findById(dictId)
                .orElseThrow(() -> new IllegalArgumentException("Dictionary not found: " + dictId));
        
        dictionary.getItems().stream()
                .filter(item -> item.getItemCode().equals(itemCode))
                .findFirst()
                .ifPresent(item -> {
                    if (itemLabel != null) item.setItemLabel(itemLabel);
                    if (sortOrder != null) item.setSortOrder(sortOrder);
                });
        
        dictionaryRepository.save(dictionary);
    }
    
    /**
     * 删除字典项
     */
    public void deleteDictionaryItem(Long dictId, String itemCode) {
        
        DataDictionary dictionary = dictionaryRepository.findById(dictId)
                .orElseThrow(() -> new IllegalArgumentException("Dictionary not found: " + dictId));
        
        dictionary.removeItem(itemCode);
        dictionaryRepository.save(dictionary);
    }
    
    // ==================== 编码规则管理 ====================
    
    /**
     * 创建编码规则
     */
    public Long createCodeRule(String ruleCode, String ruleName, String businessType,
                               String ruleExpression, Integer paddingLength,
                               Long startValue, Integer step) {
        
        if (codeRuleRepository.existsByRuleCode(ruleCode)) {
            throw new IllegalArgumentException("Rule code already exists: " + ruleCode);
        }
        
        CodeRule codeRule = new CodeRule();
        codeRule.create(ruleCode, ruleName, businessType, ruleExpression, 
                        paddingLength != null ? paddingLength : 4);
        
        if (startValue != null) {
            codeRule.setStartValue(startValue);
            codeRule.setCurrentValue(startValue);
        }
        if (step != null) {
            codeRule.setStep(step);
        }
        
        return codeRuleRepository.save(codeRule).getId();
    }
    
    /**
     * 添加编码规则要素
     */
    public void addCodeRuleElement(Long ruleId, String elementType, String elementValue, String fieldName) {
        
        CodeRule codeRule = codeRuleRepository.findById(ruleId)
                .orElseThrow(() -> new IllegalArgumentException("Code rule not found: " + ruleId));
        
        CodeRule.RuleElement.ElementType type = CodeRule.RuleElement.ElementType.valueOf(elementType);
        CodeRule.RuleElement element = codeRule.addElement(type, elementValue);
        element.setFieldName(fieldName);
        
        codeRuleRepository.save(codeRule);
    }
    
    /**
     * 生成下一个编码
     */
    public String generateCode(String businessType, Map<String, String> businessFields) {
        
        CodeRule codeRule = codeRuleRepository.findByBusinessTypeAndStatus(
                businessType, CodeRule.RuleStatus.ACTIVE)
                .orElseThrow(() -> new IllegalArgumentException("No active code rule for: " + businessType));
        
        String code = codeRule.generateNextCode();
        
        // 替换业务字段占位符
        if (businessFields != null) {
            for (Map.Entry<String, String> entry : businessFields.entrySet()) {
                code = code.replace("{" + entry.getKey() + "}", entry.getValue());
            }
        }
        
        codeRuleRepository.save(codeRule);
        
        return code;
    }
    
    /**
     * 重置编码规则流水号
     */
    public void resetCodeRuleSequence(Long ruleId) {
        
        CodeRule codeRule = codeRuleRepository.findById(ruleId)
                .orElseThrow(() -> new IllegalArgumentException("Code rule not found: " + ruleId));
        
        codeRule.resetSequence();
        codeRuleRepository.save(codeRule);
    }
}