package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.application.command.ReportDesignerService;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ReportDatasourceDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ReportTemplateDO;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/mes/report-designer")
@RequiredArgsConstructor
public class ReportDesignerController {
    
    private final ReportDesignerService service;
    
    @PostMapping("/templates")
    public ReportTemplateDO createTemplate(@RequestBody ReportTemplateDO template) {
        return service.createTemplate(template);
    }
    
    @PutMapping("/templates/{id}")
    public ReportTemplateDO updateTemplate(@PathVariable Long id, 
                                            @RequestBody ReportTemplateDO template) {
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
    public List<ReportTemplateDO> listTemplates(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String reportType) {
        return service.listTemplates(status, reportType);
    }
    
    @GetMapping("/templates/{id}")
    public ReportTemplateDO getTemplate(@PathVariable Long id) {
        return service.getTemplate(id);
    }
    
    @PostMapping("/datasources")
    public ReportDatasourceDO createDatasource(@RequestBody ReportDatasourceDO datasource) {
        return service.createDatasource(datasource);
    }
    
    @PutMapping("/datasources/{id}")
    public ReportDatasourceDO updateDatasource(@PathVariable Long id, 
                                                @RequestBody ReportDatasourceDO datasource) {
        datasource.setId(id);
        return service.updateDatasource(datasource);
    }
    
    @DeleteMapping("/datasources/{id}")
    public void deleteDatasource(@PathVariable Long id) {
        service.deleteDatasource(id);
    }
    
    @GetMapping("/datasources")
    public List<ReportDatasourceDO> listDatasources(
            @RequestParam(required = false) String datasourceType) {
        return service.listDatasources(datasourceType);
    }
    
    @GetMapping("/datasources/{id}")
    public ReportDatasourceDO getDatasource(@PathVariable Long id) {
        return service.getDatasource(id);
    }
}