package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.application.command.ConfigurationCommandService;
import com.metawebthree.mes.application.query.ConfigurationQueryService;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.config.WorkOrderType;
import com.metawebthree.mes.interfaces.dto.CodeRuleDTO;
import com.metawebthree.mes.interfaces.dto.CreateWorkOrderTypeRequest;
import com.metawebthree.mes.interfaces.dto.DataDictionaryDTO;
import com.metawebthree.mes.interfaces.dto.EntityExtensionFieldDTO;
import com.metawebthree.mes.interfaces.dto.EntityExtensionFieldValueDTO;
import com.metawebthree.mes.interfaces.dto.UpdateWorkOrderTypeRequest;
import com.metawebthree.mes.interfaces.dto.WorkOrderTypeDTO;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
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
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<List<EntityExtensionFieldDTO>> getExtensionFields(
            @RequestParam String entityType) {
        List<EntityExtensionFieldDTO> dtos = queryService.getAllExtensionFields(entityType).stream()
                .map(EntityExtensionFieldDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/extension-fields/{id}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<EntityExtensionFieldDTO> getExtensionField(@PathVariable Long id) {
        return queryService.getExtensionField(id)
                .map(field -> ResponseEntity.ok(EntityExtensionFieldDTO.fromEntity(field)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/extension-fields")
    @RequirePermission(MesPermissions.CONFIG_CREATE)
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
    @RequirePermission(MesPermissions.CONFIG_UPDATE)
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
    @RequirePermission(MesPermissions.CONFIG_DELETE)
    public ResponseEntity<Void> deleteExtensionField(@PathVariable Long id) {
        commandService.deleteExtensionField(id);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/entities/{entityType}/{entityId}/extension-values")
    @RequirePermission(MesPermissions.CONFIG_READ)
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
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<List<DataDictionaryDTO>> getAllDictionaries() {
        List<DataDictionaryDTO> dtos = queryService.getAllActiveDictionaries().stream()
                .map(DataDictionaryDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/dictionaries/{id}")
    @RequirePermission(MesPermissions.CONFIG_READ)
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
    
    @GetMapping("/dictionaries/{dictId}/items/root")
    public ResponseEntity<List<DataDictionaryDTO.DataDictionaryItemDTO>> getRootItems(
            @PathVariable Long dictId) {
        List<DataDictionaryDTO.DataDictionaryItemDTO> items = queryService.getRootDictionaryItems(dictId).stream()
                .map(DataDictionaryDTO.DataDictionaryItemDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(items);
    }
    
    @GetMapping("/dictionaries/{dictId}/items")
    public ResponseEntity<List<DataDictionaryDTO.DataDictionaryItemDTO>> getItemsByParent(
            @PathVariable Long dictId,
            @RequestParam(required = false) String parentItemCode) {
        List<DataDictionaryDTO.DataDictionaryItemDTO> items = queryService.getDictionaryItemsByParent(dictId, parentItemCode).stream()
                .map(DataDictionaryDTO.DataDictionaryItemDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(items);
    }
    
    @GetMapping("/dictionaries/code/{dictCode}/items/root")
    public ResponseEntity<List<DataDictionaryDTO.DataDictionaryItemDTO>> getRootItemsByCode(
            @PathVariable String dictCode) {
        return queryService.getDictionaryByCode(dictCode)
                .map(dict -> {
                    List<DataDictionaryDTO.DataDictionaryItemDTO> items = queryService.getRootDictionaryItems(dict.getId()).stream()
                            .map(DataDictionaryDTO.DataDictionaryItemDTO::fromEntity)
                            .collect(Collectors.toList());
                    return ResponseEntity.ok(items);
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/dictionaries/code/{dictCode}/items")
    public ResponseEntity<List<DataDictionaryDTO.DataDictionaryItemDTO>> getItemsByParentCode(
            @PathVariable String dictCode,
            @RequestParam(required = false) String parentItemCode) {
        return queryService.getDictionaryByCode(dictCode)
                .map(dict -> {
                    List<DataDictionaryDTO.DataDictionaryItemDTO> items = queryService.getDictionaryItemsByParent(dict.getId(), parentItemCode).stream()
                            .map(DataDictionaryDTO.DataDictionaryItemDTO::fromEntity)
                            .collect(Collectors.toList());
                    return ResponseEntity.ok(items);
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/dictionaries")
    @RequirePermission(MesPermissions.CONFIG_CREATE)
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
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<List<CodeRuleDTO>> getAllCodeRules() {
        List<CodeRuleDTO> dtos = queryService.getAllActiveCodeRules().stream()
                .map(CodeRuleDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/code-rules/{id}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<CodeRuleDTO> getCodeRule(@PathVariable Long id) {
        return queryService.getCodeRule(id)
                .map(rule -> ResponseEntity.ok(CodeRuleDTO.fromEntity(rule)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code-rules/business-type/{businessType}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<CodeRuleDTO> getCodeRuleByBusinessType(@PathVariable String businessType) {
        return queryService.getCodeRuleByBusinessType(businessType)
                .map(rule -> ResponseEntity.ok(CodeRuleDTO.fromEntity(rule)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/code-rules")
    @RequirePermission(MesPermissions.CONFIG_CREATE)
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
    @RequirePermission(MesPermissions.CONFIG_UPDATE)
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
    @RequirePermission(MesPermissions.CONFIG_UPDATE)
    public ResponseEntity<Void> resetCodeRuleSequence(@PathVariable Long ruleId) {
        commandService.resetCodeRuleSequence(ruleId);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/code-rules/generate/{businessType}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<Map<String, String>> generateCode(
            @PathVariable String businessType,
            @RequestParam(required = false) Map<String, String> businessFields) {
        
        String code = commandService.generateCode(businessType, businessFields);
        
        return ResponseEntity.ok(Map.of("code", code));
    }
    
    @GetMapping("/code-rules/preview/{businessType}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<Map<String, String>> previewCode(@PathVariable String businessType) {
        return queryService.previewCode(businessType)
                .map(preview -> ResponseEntity.ok(Map.of("preview", preview)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    // ==================== Work Order Type APIs ====================
    
    @GetMapping("/work-order-types")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<List<WorkOrderTypeDTO>> getAllWorkOrderTypes() {
        List<WorkOrderTypeDTO> dtos = queryService.getAllWorkOrderTypes().stream()
                .map(WorkOrderTypeDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/work-order-types/active")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<List<WorkOrderTypeDTO>> getActiveWorkOrderTypes() {
        List<WorkOrderTypeDTO> dtos = queryService.getActiveWorkOrderTypes().stream()
                .map(WorkOrderTypeDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/work-order-types/{id}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<WorkOrderTypeDTO> getWorkOrderType(@PathVariable Long id) {
        return queryService.getWorkOrderTypeById(id)
                .map(type -> ResponseEntity.ok(WorkOrderTypeDTO.fromEntity(type)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/work-order-types/code/{typeCode}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<WorkOrderTypeDTO> getWorkOrderTypeByCode(@PathVariable String typeCode) {
        return queryService.getWorkOrderTypeByCode(typeCode)
                .map(type -> ResponseEntity.ok(WorkOrderTypeDTO.fromEntity(type)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/work-order-types/default")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<WorkOrderTypeDTO> getDefaultWorkOrderType() {
        return queryService.getDefaultWorkOrderType()
                .map(type -> ResponseEntity.ok(WorkOrderTypeDTO.fromEntity(type)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/work-order-types")
    @RequirePermission(MesPermissions.CONFIG_CREATE)
    public ResponseEntity<WorkOrderTypeDTO> createWorkOrderType(@Valid @RequestBody CreateWorkOrderTypeRequest request) {
        WorkOrderType created = commandService.createWorkOrderType(
                request.getTypeCode(), request.getTypeName(), request.getDescription(), 
                request.getStatusMachineCode(), request.getProcessRouteTemplate(),
                request.getIsDefault(), request.getSortOrder());
        
        return ResponseEntity.status(201).body(WorkOrderTypeDTO.fromEntity(created));
    }
    
    @PutMapping("/work-order-types/{id}")
    @RequirePermission(MesPermissions.CONFIG_UPDATE)
    public ResponseEntity<Void> updateWorkOrderType(
            @PathVariable Long id,
            @Valid @RequestBody UpdateWorkOrderTypeRequest request) {
        
        commandService.updateWorkOrderType(id, request.getTypeName(), request.getDescription(), 
                request.getStatusMachineCode(), request.getProcessRouteTemplate(),
                request.getIsDefault(), request.getSortOrder(), request.getStatus());
        
        return ResponseEntity.ok().build();
    }
    
    @DeleteMapping("/work-order-types/{id}")
    @RequirePermission(MesPermissions.CONFIG_DELETE)
    public ResponseEntity<Void> deleteWorkOrderType(@PathVariable Long id) {
        commandService.deleteWorkOrderType(id);
        return ResponseEntity.ok().build();
    }
    
    // ==================== Product SN Rule Binding APIs ====================
    
    @GetMapping("/product-sn-rules")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<List<Map<String, Object>>> getAllProductSnRules() {
        List<Map<String, Object>> rules = queryService.getAllActiveProductSnRules().stream()
                .map(rule -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("id", rule.getId());
                    map.put("productId", rule.getProductId());
                    map.put("productCode", rule.getProductCode());
                    map.put("codeRuleId", rule.getCodeRuleId());
                    map.put("ruleCode", rule.getRuleCode());
                    map.put("isActive", rule.getIsActive());
                    map.put("description", rule.getDescription() != null ? rule.getDescription() : "");
                    return map;
                })
                .collect(Collectors.toList());
        return ResponseEntity.ok(rules);
    }
    
    @GetMapping("/product-sn-rules/product/{productId}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<Map<String, Object>> getProductSnRuleByProductId(@PathVariable Long productId) {
        return queryService.getProductSnRuleByProductId(productId)
                .map(rule -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("id", rule.getId());
                    map.put("productId", rule.getProductId());
                    map.put("productCode", rule.getProductCode());
                    map.put("codeRuleId", rule.getCodeRuleId());
                    map.put("ruleCode", rule.getRuleCode());
                    map.put("isActive", rule.getIsActive());
                    map.put("description", rule.getDescription() != null ? rule.getDescription() : "");
                    return ResponseEntity.ok(map);
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/product-sn-rules/product-code/{productCode}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<Map<String, Object>> getProductSnRuleByProductCode(@PathVariable String productCode) {
        return queryService.getProductSnRuleByProductCode(productCode)
                .map(rule -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("id", rule.getId());
                    map.put("productId", rule.getProductId());
                    map.put("productCode", rule.getProductCode());
                    map.put("codeRuleId", rule.getCodeRuleId());
                    map.put("ruleCode", rule.getRuleCode());
                    map.put("isActive", rule.getIsActive());
                    map.put("description", rule.getDescription() != null ? rule.getDescription() : "");
                    return ResponseEntity.ok(map);
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/product-sn-rules")
    @RequirePermission(MesPermissions.CONFIG_CREATE)
    public ResponseEntity<Void> bindProductSnRule(@RequestBody Map<String, Object> request) {
        Long productId = ((Number) request.get("productId")).longValue();
        String productCode = (String) request.get("productCode");
        Long codeRuleId = ((Number) request.get("codeRuleId")).longValue();
        String ruleCode = (String) request.get("ruleCode");
        String description = (String) request.get("description");
        
        commandService.bindProductSnRule(productId, productCode, codeRuleId, ruleCode, description);
        
        return ResponseEntity.status(201).build();
    }
    
    @DeleteMapping("/product-sn-rules/product/{productId}")
    @RequirePermission(MesPermissions.CONFIG_DELETE)
    public ResponseEntity<Void> unbindProductSnRule(@PathVariable Long productId) {
        commandService.unbindProductSnRule(productId);
        return ResponseEntity.ok().build();
    }
    
    @PutMapping("/product-sn-rules/product/{productId}/status")
    @RequirePermission(MesPermissions.CONFIG_UPDATE)
    public ResponseEntity<Void> updateProductSnRuleStatus(
            @PathVariable Long productId,
            @RequestBody Map<String, Boolean> request) {
        
        Boolean isActive = request.get("isActive");
        commandService.updateProductSnRuleStatus(productId, isActive);
        
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/product-sn-rules/generate/product/{productId}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<Map<String, String>> generateProductSn(
            @PathVariable Long productId,
            @RequestParam(required = false) Map<String, String> businessFields) {
        
        String sn = commandService.generateProductSn(productId, businessFields);
        
        return ResponseEntity.ok(Map.of("sn", sn));
    }
    
    @GetMapping("/product-sn-rules/generate/product-code/{productCode}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<Map<String, String>> generateProductSnByCode(
            @PathVariable String productCode,
            @RequestParam(required = false) Map<String, String> businessFields) {
        
        String sn = commandService.generateProductSnByCode(productCode, businessFields);
        
        return ResponseEntity.ok(Map.of("sn", sn));
    }
    
    @GetMapping("/code-rules/preview-sn/{businessType}")
    @RequirePermission(MesPermissions.CONFIG_READ)
    public ResponseEntity<Map<String, String>> previewSnCode(@PathVariable String businessType) {
        return queryService.previewSnCode(businessType)
                .map(preview -> ResponseEntity.ok(Map.of("preview", preview)))
                .orElse(ResponseEntity.notFound().build());
    }
}