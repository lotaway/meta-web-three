package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.application.command.ProcessFlowCommandService;
import com.metawebthree.mes.application.command.ProcessNodeTypeService;
import com.metawebthree.mes.application.query.ProcessFlowQueryService;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessFlowInstanceDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessFlowTemplateDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessFlowTemplateVersionDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessNodeTypeDO;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/mes/process-flow")
@RequiredArgsConstructor
public class ProcessFlowController {
    
    private final ProcessFlowCommandService commandService;
    private final ProcessFlowQueryService queryService;
    private final ProcessNodeTypeService nodeTypeService;
    
    // ========== 流程模板管理 ==========
    
    @PostMapping("/templates")
    public ProcessFlowTemplateDO createTemplate(@RequestBody ProcessFlowTemplateDO template) {
        return commandService.createTemplate(template);
    }
    
    @PutMapping("/templates/{id}")
    public ProcessFlowTemplateDO updateTemplate(@PathVariable Long id, 
                                                 @RequestBody ProcessFlowTemplateDO template) {
        template.setId(id);
        return commandService.updateTemplate(template);
    }
    
    @PostMapping("/templates/{id}/publish")
    public void publishTemplate(@PathVariable Long id, @RequestParam Long userId) {
        commandService.publishTemplate(id, userId);
    }
    
    @PostMapping("/templates/{id}/archive")
    public void archiveTemplate(@PathVariable Long id, @RequestParam Long userId) {
        commandService.archiveTemplate(id, userId);
    }
    
    @DeleteMapping("/templates/{id}")
    public void deleteTemplate(@PathVariable Long id) {
        commandService.deleteTemplate(id);
    }
    
    @GetMapping("/templates")
    public List<ProcessFlowTemplateDO> listTemplates(@RequestParam(required = false) String status) {
        return commandService.listTemplates(status);
    }
    
    @GetMapping("/templates/{id}")
    public ProcessFlowTemplateDO getTemplate(@PathVariable Long id) {
        return commandService.getTemplate(id);
    }
    
    // ========== 流程模板版本管理 ==========
    
    @GetMapping("/templates/{id}/versions")
    public List<ProcessFlowTemplateVersionDO> getVersionHistory(@PathVariable Long id) {
        return commandService.getVersionHistory(id);
    }
    
    @PostMapping("/templates/{id}/versions/{version}/rollback")
    public ProcessFlowTemplateDO rollbackToVersion(
            @PathVariable Long id,
            @PathVariable Integer version,
            @RequestParam Long userId) {
        return commandService.rollbackToVersion(id, version, userId);
    }
    
    @PostMapping("/templates/{id}/save-version")
    public void saveVersion(@PathVariable Long id, @RequestParam String changeDescription) {
        ProcessFlowTemplateDO template = commandService.getTemplate(id);
        if (template != null) {
            commandService.saveVersionHistory(template, changeDescription);
        }
    }
    
    // ========== 节点类型管理 ==========
    
    @GetMapping("/node-types")
    public List<ProcessNodeTypeDO> listNodeTypes(@RequestParam(required = false) String category) {
        return nodeTypeService.listNodeTypes(category);
    }
    
    @PostMapping("/node-types")
    public ProcessNodeTypeDO createNodeType(@RequestBody ProcessNodeTypeDO nodeType) {
        return nodeTypeService.createNodeType(nodeType);
    }
    
    @PutMapping("/node-types/{id}")
    public ProcessNodeTypeDO updateNodeType(@PathVariable Long id, 
                                             @RequestBody ProcessNodeTypeDO nodeType) {
        nodeType.setId(id);
        return nodeTypeService.updateNodeType(nodeType);
    }
    
    @DeleteMapping("/node-types/{id}")
    public void deleteNodeType(@PathVariable Long id) {
        nodeTypeService.deleteNodeType(id);
    }
    
    @PostMapping("/node-types/init")
    public void initDefaultNodeTypes() {
        nodeTypeService.initDefaultNodeTypes();
    }
    
    // ========== 流程实例管理 ==========
    
    @PostMapping("/instances")
    public ProcessFlowInstanceDO startInstance(@RequestBody Map<String, Object> params) {
        Long templateId = Long.valueOf(params.get("templateId").toString());
        String businessType = params.get("businessType").toString();
        String businessKey = params.get("businessKey").toString();
        Long userId = Long.valueOf(params.get("userId").toString());
        return queryService.startInstance(templateId, businessType, businessKey, userId);
    }
    
    @PostMapping("/instances/{id}/complete")
    public void completeInstance(@PathVariable Long id, @RequestParam Long userId) {
        queryService.completeInstance(id, userId);
    }
    
    @PostMapping("/instances/{id}/terminate")
    public void terminateInstance(@PathVariable Long id, @RequestParam Long userId) {
        queryService.terminateInstance(id, userId);
    }
    
    @GetMapping("/instances/{id}")
    public ProcessFlowInstanceDO getInstance(@PathVariable Long id) {
        return queryService.getInstance(id);
    }
    
    @GetMapping("/instances")
    public List<ProcessFlowInstanceDO> listInstances(
            @RequestParam(required = false) String businessType,
            @RequestParam(required = false) String status) {
        return queryService.listInstances(businessType, status);
    }
    
    @GetMapping("/instances/by-business")
    public List<ProcessFlowInstanceDO> listByBusinessKey(
            @RequestParam String businessType,
            @RequestParam String businessKey) {
        return queryService.listByBusinessKey(businessType, businessKey);
    }
}