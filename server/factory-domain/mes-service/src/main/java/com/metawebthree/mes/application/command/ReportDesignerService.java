package com.metawebthree.mes.application.command;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ReportDatasourceDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ReportTemplateDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ReportDatasourceMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.ReportTemplateMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ReportDesignerService {
    
    private final ReportTemplateMapper templateMapper;
    private final ReportDatasourceMapper datasourceMapper;
    
    // ========== 报表模板管理 ==========
    
    @Transactional
    public ReportTemplateDO createTemplate(ReportTemplateDO template) {
        template.setVersion(1);
        template.setStatus("DRAFT");
        template.setDeleted(false);
        template.setCreatedAt(LocalDateTime.now());
        templateMapper.insert(template);
        return template;
    }
    
    @Transactional
    public ReportTemplateDO updateTemplate(ReportTemplateDO template) {
        template.setUpdatedAt(LocalDateTime.now());
        templateMapper.updateById(template);
        return template;
    }
    
    @Transactional
    public void publishTemplate(Long templateId) {
        ReportTemplateDO template = templateMapper.selectById(templateId);
        if (template != null) {
            template.setStatus("PUBLISHED");
            template.setUpdatedAt(LocalDateTime.now());
            templateMapper.updateById(template);
        }
    }
    
    @Transactional
    public void deleteTemplate(Long templateId) {
        ReportTemplateDO template = templateMapper.selectById(templateId);
        if (template != null) {
            template.setDeleted(true);
            template.setUpdatedAt(LocalDateTime.now());
            templateMapper.updateById(template);
        }
    }
    
    public ReportTemplateDO getTemplate(Long templateId) {
        return templateMapper.selectById(templateId);
    }
    
    public List<ReportTemplateDO> listTemplates(String status, String reportType) {
        LambdaQueryWrapper<ReportTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReportTemplateDO::getDeleted, false);
        if (status != null && !status.isEmpty()) {
            wrapper.eq(ReportTemplateDO::getStatus, status);
        }
        if (reportType != null && !reportType.isEmpty()) {
            wrapper.eq(ReportTemplateDO::getReportType, reportType);
        }
        wrapper.orderByDesc(ReportTemplateDO::getCreatedAt);
        return templateMapper.selectList(wrapper);
    }
    
    // ========== 数据源管理 ==========
    
    @Transactional
    public ReportDatasourceDO createDatasource(ReportDatasourceDO datasource) {
        datasource.setEnabled(true);
        datasource.setDeleted(false);
        datasource.setCreatedAt(LocalDateTime.now());
        datasourceMapper.insert(datasource);
        return datasource;
    }
    
    @Transactional
    public ReportDatasourceDO updateDatasource(ReportDatasourceDO datasource) {
        datasource.setUpdatedAt(LocalDateTime.now());
        datasourceMapper.updateById(datasource);
        return datasource;
    }
    
    @Transactional
    public void deleteDatasource(Long datasourceId) {
        ReportDatasourceDO datasource = datasourceMapper.selectById(datasourceId);
        if (datasource != null) {
            datasource.setDeleted(true);
            datasource.setUpdatedAt(LocalDateTime.now());
            datasourceMapper.updateById(datasource);
        }
    }
    
    public ReportDatasourceDO getDatasource(Long datasourceId) {
        return datasourceMapper.selectById(datasourceId);
    }
    
    public List<ReportDatasourceDO> listDatasources(String datasourceType) {
        LambdaQueryWrapper<ReportDatasourceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReportDatasourceDO::getDeleted, false);
        wrapper.eq(ReportDatasourceDO::getEnabled, true);
        if (datasourceType != null && !datasourceType.isEmpty()) {
            wrapper.eq(ReportDatasourceDO::getDatasourceType, datasourceType);
        }
        wrapper.orderByDesc(ReportDatasourceDO::getCreatedAt);
        return datasourceMapper.selectList(wrapper);
    }
}