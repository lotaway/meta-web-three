package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.application.command.DashboardDesignerService;
import com.metawebthree.mes.infrastructure.persistence.dataobject.DashboardComponentDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.DashboardTemplateDO;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/mes/dashboard-designer")
@RequiredArgsConstructor
public class DashboardDesignerController {
    
    private final DashboardDesignerService service;
    
    @PostMapping("/templates")
    public DashboardTemplateDO createTemplate(@RequestBody DashboardTemplateDO template) {
        return service.createTemplate(template);
    }
    
    @PutMapping("/templates/{id}")
    public DashboardTemplateDO updateTemplate(@PathVariable Long id, 
                                               @RequestBody DashboardTemplateDO template) {
        template.setId(id);
        return service.updateTemplate(template);
    }
    
    @PostMapping("/templates/{id}/publish")
    public void publishTemplate(@PathVariable Long id) {
        service.publishTemplate(id);
    }
    
    @DeleteMapping("/templates/{id}")
    public void deleteTemplate(@PathVariable Long id) {
        service.deleteTemplate(id);
    }
    
    @GetMapping("/templates")
    public List<DashboardTemplateDO> listTemplates(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String templateType) {
        return service.listTemplates(status, templateType);
    }
    
    @GetMapping("/templates/{id}")
    public DashboardTemplateDO getTemplate(@PathVariable Long id) {
        return service.getTemplate(id);
    }
    
    @PostMapping("/components")
    public DashboardComponentDO createComponent(@RequestBody DashboardComponentDO component) {
        return service.createComponent(component);
    }
    
    @PutMapping("/components/{id}")
    public DashboardComponentDO updateComponent(@PathVariable Long id, 
                                                 @RequestBody DashboardComponentDO component) {
        component.setId(id);
        return service.updateComponent(component);
    }
    
    @DeleteMapping("/components/{id}")
    public void deleteComponent(@PathVariable Long id) {
        service.deleteComponent(id);
    }
    
    @GetMapping("/components")
    public List<DashboardComponentDO> listComponents(
            @RequestParam(required = false) String componentType) {
        return service.listComponents(componentType);
    }
    
    @GetMapping("/components/{id}")
    public DashboardComponentDO getComponent(@PathVariable Long id) {
        return service.getComponent(id);
    }
    
    @PostMapping("/components/init")
    public void initDefaultComponents() {
        service.initDefaultComponents();
    }
}