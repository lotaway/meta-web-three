package com.metawebthree.mes.application.query;

import com.metawebthree.mes.domain.config.WorkOrderType;
import com.metawebthree.mes.domain.entity.CodeRule;
import com.metawebthree.mes.domain.entity.DataDictionary;
import com.metawebthree.mes.domain.entity.EntityExtensionField;
import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import com.metawebthree.mes.domain.repository.CodeRuleRepository;
import com.metawebthree.mes.domain.repository.DataDictionaryRepository;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldRepository;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldValueRepository;
import com.metawebthree.mes.domain.repository.WorkOrderTypeRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class ConfigurationQueryService {
    
    private final EntityExtensionFieldRepository fieldRepository;
    private final EntityExtensionFieldValueRepository fieldValueRepository;
    private final DataDictionaryRepository dictionaryRepository;
    private final CodeRuleRepository codeRuleRepository;
    private final WorkOrderTypeRepository workOrderTypeRepository;
    
    public ConfigurationQueryService(
            EntityExtensionFieldRepository fieldRepository,
            EntityExtensionFieldValueRepository fieldValueRepository,
            DataDictionaryRepository dictionaryRepository,
            CodeRuleRepository codeRuleRepository,
            WorkOrderTypeRepository workOrderTypeRepository) {
        this.fieldRepository = fieldRepository;
        this.fieldValueRepository = fieldValueRepository;
        this.dictionaryRepository = dictionaryRepository;
        this.codeRuleRepository = codeRuleRepository;
        this.workOrderTypeRepository = workOrderTypeRepository;
    }
    
    public List<EntityExtensionField> getAllExtensionFields(String entityType) {
        return fieldRepository.findByEntityType(entityType);
    }
    
    public Optional<EntityExtensionField> getExtensionField(Long id) {
        return fieldRepository.findById(id);
    }
    
    public List<EntityExtensionFieldValue> getExtensionFieldValues(String entityType, Long entityId) {
        return fieldValueRepository.findByEntityTypeAndEntityId(entityType, entityId);
    }
    
    public Map<String, String> getExtensionFieldValuesAsMap(String entityType, Long entityId) {
        return fieldValueRepository.findByEntityTypeAndEntityId(entityType, entityId).stream()
                .collect(Collectors.toMap(
                        EntityExtensionFieldValue::getFieldCode,
                        EntityExtensionFieldValue::getFieldValue,
                        (v1, v2) -> v2
                ));
    }
    
    public List<DataDictionary> getAllActiveDictionaries() {
        return dictionaryRepository.findAllActive();
    }
    
    public Optional<DataDictionary> getDictionaryByCode(String dictCode) {
        return dictionaryRepository.findByDictCode(dictCode);
    }
    
    public Optional<DataDictionary> getDictionary(Long id) {
        return dictionaryRepository.findById(id);
    }
    
    public List<DataDictionary.DataDictionaryItem> getDictionaryItemsByParent(Long dictId, String parentItemCode) {
        return dictionaryRepository.findItemsByDictIdAndParentItemCode(dictId, parentItemCode);
    }
    
    public List<DataDictionary.DataDictionaryItem> getRootDictionaryItems(Long dictId) {
        return dictionaryRepository.findRootItemsByDictId(dictId);
    }
    
    public Optional<CodeRule> getCodeRule(Long id) {
        return codeRuleRepository.findById(id);
    }
    
    public Optional<CodeRule> getCodeRuleByCode(String ruleCode) {
        return codeRuleRepository.findByRuleCode(ruleCode);
    }
    
    public Optional<CodeRule> getCodeRuleByBusinessType(String businessType) {
        return codeRuleRepository.findByBusinessTypeAndStatus(businessType, CodeRule.RuleStatus.ACTIVE);
    }
    
    public List<CodeRule> getAllActiveCodeRules() {
        return codeRuleRepository.findAllActive();
    }
    
    public Optional<String> previewCode(String businessType) {
        return getCodeRuleByBusinessType(businessType)
                .map(this::generatePreview);
    }
    
    private String generatePreview(CodeRule codeRule) {
        StringBuilder preview = new StringBuilder();
        for (CodeRule.RuleElement element : codeRule.getElements()) {
            switch (element.getType()) {
                case PREFIX -> preview.append(element.getValue());
                case DATE -> preview.append(formatDatePreview(element.getValue()));
                case SEQUENCE -> preview.append(String.format("%0" + codeRule.getPaddingLength() + "d", 1));
                case BUSINESS_FIELD -> preview.append("{").append(element.getFieldName()).append("}");
                case DELIMITER -> preview.append(element.getValue());
            }
        }
        return preview.toString();
    }
    
    private String formatDatePreview(String pattern) {
        LocalDate now = LocalDate.now();
        return pattern
                .replace("YYYY", String.valueOf(now.getYear()))
                .replace("YY", String.format("%02d", now.getYear() % 100))
                .replace("MM", String.format("%02d", now.getMonthValue()))
                .replace("DD", String.format("%02d", now.getDayOfMonth()));
    }
    
    // ==================== Work Order Type Queries ====================
    
    public List<WorkOrderType> getAllWorkOrderTypes() {
        return workOrderTypeRepository.findAll();
    }
    
    public Optional<WorkOrderType> getWorkOrderTypeById(Long id) {
        return workOrderTypeRepository.findById(id);
    }
    
    public Optional<WorkOrderType> getWorkOrderTypeByCode(String typeCode) {
        return workOrderTypeRepository.findByTypeCode(typeCode);
    }
    
    public Optional<WorkOrderType> getDefaultWorkOrderType() {
        return workOrderTypeRepository.findByIsDefaultTrue();
    }
    
    public List<WorkOrderType> getActiveWorkOrderTypes() {
        return workOrderTypeRepository.findByStatus("ACTIVE");
    }
}