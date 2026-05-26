package com.metawebthree.mes.application.query;

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
 * 配置管理查询服务
 * 负责查询扩展字段、数据字典、编码规则
 */
@Service
public class ConfigurationQueryService {
    
    private final EntityExtensionFieldRepository fieldRepository;
    private final EntityExtensionFieldValueRepository fieldValueRepository;
    private final DataDictionaryRepository dictionaryRepository;
    private final CodeRuleRepository codeRuleRepository;
    
    public ConfigurationQueryService(
            EntityExtensionFieldRepository fieldRepository,
            EntityExtensionFieldValueRepository fieldValueRepository,
            DataDictionaryRepository dictionaryRepository,
            CodeRuleRepository codeRuleRepository) {
        this.fieldRepository = fieldRepository;
        this.fieldValueRepository = fieldValueRepository;
        this.dictionaryRepository = dictionaryRepository;
        this.codeRuleRepository = codeRuleRepository;
    }
    
    // ==================== 扩展字段查询 ====================
    
    /**
     * 获取所有扩展字段定义
     */
    public List<EntityExtensionField> getAllExtensionFields(String entityType) {
        return fieldRepository.findByEntityType(entityType);
    }
    
    /**
     * 获取扩展字段定义详情
     */
    public EntityExtensionField getExtensionField(Long id) {
        return fieldRepository.findById(id);
    }
    
    /**
     * 获取实体扩展字段值
     */
    public List<EntityExtensionFieldValue> getExtensionFieldValues(String entityType, Long entityId) {
        return fieldValueRepository.findByEntityTypeAndEntityId(entityType, entityId);
    }
    
    /**
     * 获取扩展字段值（转换为Map）
     */
    public Map<String, String> getExtensionFieldValuesAsMap(String entityType, Long entityId) {
        return fieldValueRepository.findByEntityTypeAndEntityId(entityType, entityId).stream()
                .collect(Collectors.toMap(
                        EntityExtensionFieldValue::getFieldCode,
                        EntityExtensionFieldValue::getFieldValue,
                        (v1, v2) -> v2
                ));
    }
    
    // ==================== 数据字典查询 ====================
    
    /**
     * 获取所有启用的数据字典
     */
    public List<DataDictionary> getAllActiveDictionaries() {
        return dictionaryRepository.findAllActive();
    }
    
    /**
     * 根据字典编码获取数据字典
     */
    public DataDictionary getDictionaryByCode(String dictCode) {
        return dictionaryRepository.findByDictCode(dictCode).orElse(null);
    }
    
    /**
     * 获取数据字典详情（含字典项）
     */
    public DataDictionary getDictionary(Long id) {
        return dictionaryRepository.findById(id).orElse(null);
    }
    
    // ==================== 编码规则查询 ====================
    
    /**
     * 获取编码规则
     */
    public CodeRule getCodeRule(Long id) {
        return codeRuleRepository.findById(id).orElse(null);
    }
    
    /**
     * 根据规则编码获取编码规则
     */
    public CodeRule getCodeRuleByCode(String ruleCode) {
        return codeRuleRepository.findByRuleCode(ruleCode).orElse(null);
    }
    
    /**
     * 根据业务类型获取编码规则
     */
    public CodeRule getCodeRuleByBusinessType(String businessType) {
        return codeRuleRepository.findByBusinessTypeAndStatus(
                businessType, CodeRule.RuleStatus.ACTIVE).orElse(null);
    }
    
    /**
     * 预生成编码预览（不实际生成）
     */
    public String previewCode(String businessType) {
        CodeRule codeRule = getCodeRuleByBusinessType(businessType);
        if (codeRule == null) {
            return null;
        }
        
        StringBuilder preview = new StringBuilder();
        for (CodeRule.RuleElement element : codeRule.getElements()) {
            switch (element.getType()) {
                case PREFIX:
                    preview.append(element.getValue());
                    break;
                case DATE:
                    preview.append(formatDatePreview(element.getValue()));
                    break;
                case SEQUENCE:
                    preview.append(String.format("%0" + codeRule.getPaddingLength() + "d", 1));
                    break;
                case BUSINESS_FIELD:
                    preview.append("{").append(element.getFieldName()).append("}");
                    break;
                case DELIMITER:
                    preview.append(element.getValue());
                    break;
            }
        }
        return preview.toString();
    }
    
    private String formatDatePreview(String pattern) {
        String result = pattern;
        result = result.replace("YYYY", "2026");
        result = result.replace("YY", "26");
        result = result.replace("MM", "05");
        result = result.replace("DD", "26");
        return result;
    }
}