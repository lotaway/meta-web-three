package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.mes.application.command.ConfigurationCommandService;
import com.metawebthree.mes.application.query.ConfigurationQueryService;
import com.metawebthree.mes.domain.entity.CodeRule;
import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import com.metawebthree.mes.interfaces.dto.CodeRuleDTO;
import com.metawebthree.mes.interfaces.dto.DataDictionaryDTO;
import com.metawebthree.mes.interfaces.dto.EntityExtensionFieldDTO;
import com.metawebthree.mes.interfaces.dto.EntityExtensionFieldValueDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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
    
    @GetMapping("/extension-fields")
    public ResponseEntity<List<EntityExtensionFieldDTO>> getExtensionFields(
            @RequestParam String entityType) {
        List<EntityExtensionFieldDTO> dtos = queryService.getAllExtensionFields(entityType).stream()
                .map(EntityExtensionFieldDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/extension-fields/{id}")
    public ResponseEntity<EntityExtensionFieldDTO> getExtensionField(@PathVariable Long id) {
        return queryService.getExtensionField(id)
                .map(field -> ResponseEntity.ok(EntityExtensionFieldDTO.fromEntity(field)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/extension-fields")
    public ResponseEntity<Void> createExtensionField(@RequestBody Map<String, Object> request) {
        String entityType = (String) request.get("entityType");
        String fieldCode = (String) request.get("fieldCode");
        String fieldName = (String) request.get("fieldName");
        String fieldType = (String) request.get("fieldType");
        String fieldGroup = (String) request.get("fieldGroup");
        Boolean required = (Boolean) request.get("required");
        String defaultValue = (String) request.get("defaultValue");
        String validationRule = (String) request.get("validationRule");
        
        commandService.createExtensionField(
                entityType, fieldCode, fieldName, fieldType, fieldGroup,
                required, defaultValue, validationRule);
        
        return ResponseEntity.status(201).build();
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
    public ResponseEntity<List<EntityExtensionFieldValueDTO>> getExtensionFieldValues(
            @PathVariable String entityType,
            @PathVariable Long entityId) {
        List<EntityExtensionFieldValueDTO> dtos = queryService.getExtensionFieldValues(entityType, entityId).stream()
                .map(EntityExtensionFieldValueDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping("/entities/{entityType}/{entityId}/extension-values")
    public ResponseEntity<Void> setExtensionFieldValue(
            @PathVariable String entityType,
            @PathVariable Long entityId,
            @RequestBody Map<String, String> fieldValues) {
        
        commandService.setExtensionFieldValues(entityType, entityId, fieldValues);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/dictionaries")
    public ResponseEntity<List<DataDictionaryDTO>> getAllDictionaries() {
        List<DataDictionaryDTO> dtos = queryService.getAllActiveDictionaries().stream()
                .map(DataDictionaryDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/dictionaries/{id}")
    public ResponseEntity<DataDictionaryDTO> getDictionary(@PathVariable Long id) {
        return queryService.getDictionary(id)
                .map(dict -> ResponseEntity.ok(DataDictionaryDTO.fromEntity(dict)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/dictionaries/code/{dictCode}")
    public ResponseEntity<DataDictionaryDTO> getDictionaryByCode(@PathVariable String dictCode) {
        return queryService.getDictionaryByCode(dictCode)
                .map(dict -> ResponseEntity.ok(DataDictionaryDTO.fromEntity(dict)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/dictionaries")
    public ResponseEntity<Void> createDictionary(@RequestBody Map<String, Object> request) {
        String dictCode = (String) request.get("dictCode");
        String dictName = (String) request.get("dictName");
        String description = (String) request.get("description");
        
        commandService.createDataDictionary(dictCode, dictName, description);
        
        return ResponseEntity.status(201).build();
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
    
    @GetMapping("/code-rules")
    public ResponseEntity<List<CodeRuleDTO>> getAllCodeRules() {
        List<CodeRuleDTO> dtos = queryService.getAllActiveCodeRules().stream()
                .map(CodeRuleDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/code-rules/{id}")
    public ResponseEntity<CodeRuleDTO> getCodeRule(@PathVariable Long id) {
        return queryService.getCodeRule(id)
                .map(rule -> ResponseEntity.ok(CodeRuleDTO.fromEntity(rule)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code-rules/business-type/{businessType}")
    public ResponseEntity<CodeRuleDTO> getCodeRuleByBusinessType(@PathVariable String businessType) {
        return queryService.getCodeRuleByBusinessType(businessType)
                .map(rule -> ResponseEntity.ok(CodeRuleDTO.fromEntity(rule)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/code-rules")
    public ResponseEntity<Void> createCodeRule(@RequestBody Map<String, Object> request) {
        String ruleCode = (String) request.get("ruleCode");
        String ruleName = (String) request.get("ruleName");
        String businessType = (String) request.get("businessType");
        String ruleExpression = (String) request.get("ruleExpression");
        Integer paddingLength = (Integer) request.get("paddingLength");
        Long startValue = request.get("startValue") != null ?
                ((Number) request.get("startValue")).longValue() : null;
        Integer step = (Integer) request.get("step");
        
        commandService.createCodeRule(
                ruleCode, ruleName, businessType, ruleExpression,
                paddingLength, startValue, step);
        
        return ResponseEntity.status(201).build();
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
        return queryService.previewCode(businessType)
                .map(preview -> ResponseEntity.ok(Map.of("preview", preview)))
                .orElse(ResponseEntity.notFound().build());
    }
}