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

import java.util.Map;

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
    
    public Long createExtensionField(String entityType, String fieldCode, String fieldName,
                                     String fieldType, String fieldGroup, Boolean required,
                                     String defaultValue, String validationRule) {
        
        if (fieldRepository.existsByEntityTypeAndFieldCode(entityType, fieldCode)) {
            throw new IllegalArgumentException("Field code already exists: " + fieldCode);
        }
        
        EntityExtensionField.FieldType type = EntityExtensionField.FieldType.valueOf(fieldType);
        EntityExtensionField field = EntityExtensionField.create(entityType, fieldCode, fieldName, type, fieldGroup);
        field.setRequired(required != null ? required : false);
        field.setDefaultValue(defaultValue);
        field.setValidationRule(validationRule);
        
        return fieldRepository.save(field).getId();
    }
    
    public void updateExtensionField(Long id, String fieldName, String fieldGroup,
                                     Boolean required, String defaultValue, String validationRule,
                                     Boolean listVisible, Boolean searchable) {
        
        EntityExtensionField field = fieldRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Extension field not found: " + id));
        
        if (fieldName != null) field.setFieldName(fieldName);
        if (fieldGroup != null) field.setFieldGroup(fieldGroup);
        if (required != null) field.setRequired(required);
        if (defaultValue != null) field.setDefaultValue(defaultValue);
        if (validationRule != null) field.setValidationRule(validationRule);
        if (listVisible != null) field.setListVisible(listVisible);
        if (searchable != null) field.setSearchable(searchable);
        
        fieldRepository.save(field);
    }
    
    public void deleteExtensionField(Long id) {
        fieldRepository.delete(id);
    }
    
    public void setExtensionFieldValue(String entityType, Long entityId,
                                       String fieldCode, String fieldValue) {
        EntityExtensionField field = fieldRepository.findByEntityTypeAndFieldCode(entityType, fieldCode)
                .orElseThrow(() -> new IllegalArgumentException("Extension field not defined: " + fieldCode));
        
        if (!field.validateValue(fieldValue)) {
            throw new IllegalArgumentException("Field value validation failed: " + fieldCode);
        }
        
        EntityExtensionFieldValue value = new EntityExtensionFieldValue();
        value.create(entityType, entityId, fieldCode, fieldValue);
        fieldValueRepository.save(value);
    }
    
    public void setExtensionFieldValues(String entityType, Long entityId,
                                        Map<String, String> fieldValues) {
        for (Map.Entry<String, String> entry : fieldValues.entrySet()) {
            setExtensionFieldValue(entityType, entityId, entry.getKey(), entry.getValue());
        }
    }
    
    public Long createDataDictionary(String dictCode, String dictName, String description) {
        
        if (dictionaryRepository.existsByDictCode(dictCode)) {
            throw new IllegalArgumentException("Dictionary code already exists: " + dictCode);
        }
        
        DataDictionary dictionary = DataDictionary.create(dictCode, dictName, description);
        
        return dictionaryRepository.save(dictionary).getId();
    }
    
    public void addDictionaryItem(Long dictId, String itemCode, String itemLabel,
                                  String parentItemCode, Integer sortOrder) {
        
        DataDictionary dictionary = dictionaryRepository.findById(dictId)
                .orElseThrow(() -> new IllegalArgumentException("Dictionary not found: " + dictId));
        
        DataDictionary.DataDictionaryItem item = dictionary.addItem(itemCode, itemLabel, sortOrder);
        item.setParentItemCode(parentItemCode);
        
        dictionaryRepository.save(dictionary);
    }
    
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
    
    public void deleteDictionaryItem(Long dictId, String itemCode) {
        
        DataDictionary dictionary = dictionaryRepository.findById(dictId)
                .orElseThrow(() -> new IllegalArgumentException("Dictionary not found: " + dictId));
        
        dictionary.removeItem(itemCode);
        dictionaryRepository.save(dictionary);
    }
    
    public Long createCodeRule(String ruleCode, String ruleName, String businessType,
                               String ruleExpression, Integer paddingLength,
                               Long startValue, Integer step) {
        
        if (codeRuleRepository.existsByRuleCode(ruleCode)) {
            throw new IllegalArgumentException("Rule code already exists: " + ruleCode);
        }
        
        CodeRule codeRule = CodeRule.create(ruleCode, ruleName, businessType, ruleExpression,
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
    
    public void addCodeRuleElement(Long ruleId, String elementType, String elementValue, String fieldName) {
        
        CodeRule codeRule = codeRuleRepository.findById(ruleId)
                .orElseThrow(() -> new IllegalArgumentException("Code rule not found: " + ruleId));
        
        CodeRule.RuleElement.ElementType type = CodeRule.RuleElement.ElementType.valueOf(elementType);
        CodeRule.RuleElement element = codeRule.addElement(type, elementValue);
        element.setFieldName(fieldName);
        
        codeRuleRepository.save(codeRule);
    }
    
    public String generateCode(String businessType, Map<String, String> businessFields) {
        
        CodeRule codeRule = codeRuleRepository.findByBusinessTypeAndStatus(
                businessType, CodeRule.RuleStatus.ACTIVE)
                .orElseThrow(() -> new IllegalArgumentException("No active code rule for: " + businessType));
        
        String code = codeRule.generateNextCode();
        
        if (businessFields != null) {
            for (Map.Entry<String, String> entry : businessFields.entrySet()) {
                code = code.replace("{" + entry.getKey() + "}", entry.getValue());
            }
        }
        
        codeRuleRepository.save(codeRule);
        
        return code;
    }
    
    public void resetCodeRuleSequence(Long ruleId) {
        
        CodeRule codeRule = codeRuleRepository.findById(ruleId)
                .orElseThrow(() -> new IllegalArgumentException("Code rule not found: " + ruleId));
        
        codeRule.resetSequence();
        codeRuleRepository.save(codeRule);
    }
}