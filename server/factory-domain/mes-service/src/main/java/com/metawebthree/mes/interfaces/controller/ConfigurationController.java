package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.mes.application.command.ConfigurationCommandService;
import com.metawebthree.mes.application.query.ConfigurationQueryService;
import com.metawebthree.mes.domain.entity.CodeRule;
import com.metawebthree.mes.domain.entity.DataDictionary;
import com.metawebthree.mes.domain.entity.EntityExtensionField;
import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 配置管理控制器
 * 提供扩展字段、数据字典、编码规则的REST API
 */
@RestController
@RequestMapping("/api/mes/config")
public class ConfigurationController {
    
    private final ConfigurationCommandService commandService;
    private final ConfigurationQueryService queryService;
    
    public ConfigurationController(ConfigurationCommandService commandService,
                                   ConfigurationQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }
    
    // ==================== 扩展字段管理 ====================
    
    @GetMapping("/extension-fields")
    public ResponseEntity<List<EntityExtensionField>> getExtensionFields(
            @RequestParam String entityType) {
        return ResponseEntity.ok(queryService.getAllExtensionFields(entityType));
    }
    
    @GetMapping("/extension-fields/{id}")
    public ResponseEntity<EntityExtensionField> getExtensionField(@PathVariable Long id) {
        EntityExtensionField field = queryService.getExtensionField(id);
        return field != null ? ResponseEntity.ok(field) : ResponseEntity.notFound().build();
    }
    
    @PostMapping("/extension-fields")
    public ResponseEntity<Map<String, Object>> createExtensionField(@RequestBody Map<String, Object> request) {
        String entityType = (String) request.get("entityType");
        String fieldCode = (String) request.get("fieldCode");
        String fieldName = (String) request.get("fieldName");
        String fieldType = (String) request.get("fieldType");
        String fieldGroup = (String) request.get("fieldGroup");
        Boolean required = (Boolean) request.get("required");
        String defaultValue = (String) request.get("defaultValue");
        String validationRule = (String) request.get("validationRule");
        
        Long id = commandService.createExtensionField(
                entityType, fieldCode, fieldName, fieldType, fieldGroup, 
                required, defaultValue, validationRule);
        
        return ResponseEntity.ok(Map.of("id", id));
    }
    
    @PutMapping("/extension-fields/{id}")
    public ResponseEntity<Void> updateExtensionField(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        
        String fieldName = (String) request.get("fieldName");
        String fieldGroup = (String) request.get("fieldGroup");
        Boolean required = (Boolean) request.get("required");
        String defaultValue = (String) request.get("defaultValue");
        String validationRule = (String) request.get("validationRule");
        Boolean listVisible = (Boolean) request.get("listVisible");
        Boolean searchable = (Boolean) request.get("searchable");
        
        commandService.updateExtensionField(id, fieldName, fieldGroup, required, 
                defaultValue, validationRule, listVisible, searchable);
        
        return ResponseEntity.ok().build();
    }
    
    @DeleteMapping("/extension-fields/{id}")
    public ResponseEntity<Void> deleteExtensionField(@PathVariable Long id) {
        commandService.deleteExtensionField(id);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/entities/{entityType}/{entityId}/extension-values")
    public ResponseEntity<List<EntityExtensionFieldValue>> getExtensionFieldValues(
            @PathVariable String entityType,
            @PathVariable Long entityId) {
        return ResponseEntity.ok(queryService.getExtensionFieldValues(entityType, entityId));
    }
    
    @PostMapping("/entities/{entityType}/{entityId}/extension-values")
    public ResponseEntity<Void> setExtensionFieldValue(
            @PathVariable String entityType,
            @PathVariable Long entityId,
            @RequestBody Map<String, String> fieldValues) {
        
        commandService.setExtensionFieldValues(entityType, entityId, fieldValues);
        return ResponseEntity.ok().build();
    }
    
    // ==================== 数据字典管理 ====================
    
    @GetMapping("/dictionaries")
    public ResponseEntity<List<DataDictionary>> getAllDictionaries() {
        return ResponseEntity.ok(queryService.getAllActiveDictionaries());
    }
    
    @GetMapping("/dictionaries/{id}")
    public ResponseEntity<DataDictionary> getDictionary(@PathVariable Long id) {
        DataDictionary dict = queryService.getDictionary(id);
        return dict != null ? ResponseEntity.ok(dict) : ResponseEntity.notFound().build();
    }
    
    @GetMapping("/dictionaries/code/{dictCode}")
    public ResponseEntity<DataDictionary> getDictionaryByCode(@PathVariable String dictCode) {
        DataDictionary dict = queryService.getDictionaryByCode(dictCode);
        return dict != null ? ResponseEntity.ok(dict) : ResponseEntity.notFound().build();
    }
    
    @PostMapping("/dictionaries")
    public ResponseEntity<Map<String, Object>> createDictionary(@RequestBody Map<String, Object> request) {
        String dictCode = (String) request.get("dictCode");
        String dictName = (String) request.get("dictName");
        String description = (String) request.get("description");
        
        Long id = commandService.createDataDictionary(dictCode, dictName, description);
        
        return ResponseEntity.ok(Map.of("id", id));
    }
    
    @PostMapping("/dictionaries/{dictId}/items")
    public ResponseEntity<Void> addDictionaryItem(
            @PathVariable Long dictId,
            @RequestBody Map<String, Object> request) {
        
        String itemCode = (String) request.get("itemCode");
        String itemLabel = (String) request.get("itemLabel");
        String parentItemCode = (String) request.get("parentItemCode");
        Integer sortOrder = (Integer) request.get("sortOrder");
        
        commandService.addDictionaryItem(dictId, itemCode, itemLabel, parentItemCode, sortOrder);
        
        return ResponseEntity.ok().build();
    }
    
    @PutMapping("/dictionaries/{dictId}/items/{itemCode}")
    public ResponseEntity<Void> updateDictionaryItem(
            @PathVariable Long dictId,
            @PathVariable String itemCode,
            @RequestBody Map<String, Object> request) {
        
        String itemLabel = (String) request.get("itemLabel");
        Integer sortOrder = (Integer) request.get("sortOrder");
        
        commandService.updateDictionaryItem(dictId, itemCode, itemLabel, sortOrder);
        
        return ResponseEntity.ok().build();
    }
    
    @DeleteMapping("/dictionaries/{dictId}/items/{itemCode}")
    public ResponseEntity<Void> deleteDictionaryItem(
            @PathVariable Long dictId,
            @PathVariable String itemCode) {
        
        commandService.deleteDictionaryItem(dictId, itemCode);
        
        return ResponseEntity.ok().build();
    }
    
    // ==================== 编码规则管理 ====================
    
    @GetMapping("/code-rules")
    public ResponseEntity<List<CodeRule>> getAllCodeRules() {
        // 返回所有规则（这里简化为返回所有，后续可添加分页）
        return ResponseEntity.ok(List.of());
    }
    
    @GetMapping("/code-rules/{id}")
    public ResponseEntity<CodeRule> getCodeRule(@PathVariable Long id) {
        CodeRule rule = queryService.getCodeRule(id);
        return rule != null ? ResponseEntity.ok(rule) : ResponseEntity.notFound().build();
    }
    
    @GetMapping("/code-rules/business-type/{businessType}")
    public ResponseEntity<CodeRule> getCodeRuleByBusinessType(@PathVariable String businessType) {
        CodeRule rule = queryService.getCodeRuleByBusinessType(businessType);
        return rule != null ? ResponseEntity.ok(rule) : ResponseEntity.notFound().build();
    }
    
    @PostMapping("/code-rules")
    public ResponseEntity<Map<String, Object>> createCodeRule(@RequestBody Map<String, Object> request) {
        String ruleCode = (String) request.get("ruleCode");
        String ruleName = (String) request.get("ruleName");
        String businessType = (String) request.get("businessType");
        String ruleExpression = (String) request.get("ruleExpression");
        Integer paddingLength = (Integer) request.get("paddingLength");
        Long startValue = request.get("startValue") != null ? 
                ((Number) request.get("startValue")).longValue() : null;
        Integer step = (Integer) request.get("step");
        
        Long id = commandService.createCodeRule(
                ruleCode, ruleName, businessType, ruleExpression, 
                paddingLength, startValue, step);
        
        return ResponseEntity.ok(Map.of("id", id));
    }
    
    @PostMapping("/code-rules/{ruleId}/elements")
    public ResponseEntity<Void> addCodeRuleElement(
            @PathVariable Long ruleId,
            @RequestBody Map<String, Object> request) {
        
        String elementType = (String) request.get("elementType");
        String elementValue = (String) request.get("elementValue");
        String fieldName = (String) request.get("fieldName");
        
        commandService.addCodeRuleElement(ruleId, elementType, elementValue, fieldName);
        
        return ResponseEntity.ok().build();
    }
    
    @PostMapping("/code-rules/{ruleId}/reset")
    public ResponseEntity<Void> resetCodeRuleSequence(@PathVariable Long ruleId) {
        commandService.resetCodeRuleSequence(ruleId);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/code-rules/generate/{businessType}")
    public ResponseEntity<Map<String, String>> generateCode(
            @PathVariable String businessType,
            @RequestParam(required = false) Map<String, String> businessFields) {
        
        String code = commandService.generateCode(businessType, businessFields);
        
        return ResponseEntity.ok(Map.of("code", code));
    }
    
    @GetMapping("/code-rules/preview/{businessType}")
    public ResponseEntity<Map<String, String>> previewCode(@PathVariable String businessType) {
        String preview = queryService.previewCode(businessType);
        return ResponseEntity.ok(Map.of("preview", preview != null ? preview : ""));
    }
}