package com.metawebthree.mes.application.command;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.DashboardComponentDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.DashboardTemplateDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.DashboardComponentMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.DashboardTemplateMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class DashboardDesignerService {
    
    private final DashboardTemplateMapper templateMapper;
    private final DashboardComponentMapper componentMapper;
    
    @Transactional
    public DashboardTemplateDO createTemplate(DashboardTemplateDO template) {
        template.setStatus("DRAFT");
        template.setRefreshInterval(30);
        template.setDeleted(false);
        template.setCreatedAt(LocalDateTime.now());
        templateMapper.insert(template);
        return template;
    }
    
    @Transactional
    public DashboardTemplateDO updateTemplate(DashboardTemplateDO template) {
        template.setUpdatedAt(LocalDateTime.now());
        templateMapper.updateById(template);
        return template;
    }
    
    @Transactional
    public void publishTemplate(Long templateId) {
        DashboardTemplateDO template = templateMapper.selectById(templateId);
        if (template != null) {
            template.setStatus("PUBLISHED");
            template.setUpdatedAt(LocalDateTime.now());
            templateMapper.updateById(template);
        }
    }
    
    @Transactional
    public void deleteTemplate(Long templateId) {
        DashboardTemplateDO template = templateMapper.selectById(templateId);
        if (template != null) {
            template.setDeleted(true);
            template.setUpdatedAt(LocalDateTime.now());
            templateMapper.updateById(template);
        }
    }
    
    public DashboardTemplateDO getTemplate(Long templateId) {
        return templateMapper.selectById(templateId);
    }
    
    public List<DashboardTemplateDO> listTemplates(String status, String templateType) {
        LambdaQueryWrapper<DashboardTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DashboardTemplateDO::getDeleted, false);
        if (status != null && !status.isEmpty()) {
            wrapper.eq(DashboardTemplateDO::getStatus, status);
        }
        if (templateType != null && !templateType.isEmpty()) {
            wrapper.eq(DashboardTemplateDO::getTemplateType, templateType);
        }
        wrapper.orderByDesc(DashboardTemplateDO::getCreatedAt);
        return templateMapper.selectList(wrapper);
    }
    
    @Transactional
    public DashboardComponentDO createComponent(DashboardComponentDO component) {
        component.setEnabled(true);
        component.setDeleted(false);
        component.setCreatedAt(LocalDateTime.now());
        componentMapper.insert(component);
        return component;
    }
    
    @Transactional
    public DashboardComponentDO updateComponent(DashboardComponentDO component) {
        componentMapper.updateById(component);
        return component;
    }
    
    @Transactional
    public void deleteComponent(Long componentId) {
        DashboardComponentDO component = componentMapper.selectById(componentId);
        if (component != null) {
            component.setDeleted(true);
            componentMapper.updateById(component);
        }
    }
    
    public DashboardComponentDO getComponent(Long componentId) {
        return componentMapper.selectById(componentId);
    }
    
    public List<DashboardComponentDO> listComponents(String componentType) {
        LambdaQueryWrapper<DashboardComponentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DashboardComponentDO::getDeleted, false)
               .eq(DashboardComponentDO::getEnabled, true);
        if (componentType != null && !componentType.isEmpty()) {
            wrapper.eq(DashboardComponentDO::getComponentType, componentType);
        }
        wrapper.orderByAsc(DashboardComponentDO::getSortOrder);
        return componentMapper.selectList(wrapper);
    }
    
    @Transactional
    public void initDefaultComponents() {
        if (componentMapper.selectCount(new LambdaQueryWrapper<>()) > 0) {
            return;
        }
        
        List<DashboardComponentDO> defaults = List.of(
            createComponentDO("kpi-card", "KPI卡片", "CHART", "KPI指标展示", 1),
            createComponentDO("line-chart", "折线图", "CHART", "趋势展示", 2),
            createComponentDO("bar-chart", "柱状图", "CHART", "对比分析", 3),
            createComponentDO("pie-chart", "饼图", "CHART", "占比分析", 4),
            createComponentDO("gauge", "仪表盘", "CHART", "达成率展示", 5),
            createComponentDO("data-table", "数据表格", "TABLE", "明细数据展示", 6),
            createComponentDO("process-flow", "流程图", "DIAGRAM", "流程状态展示", 7),
            createComponentDO("status-indicator", "状态指示灯", "INDICATOR", "设备状态监控", 8)
        );
        
        for (DashboardComponentDO comp : defaults) {
            componentMapper.insert(comp);
        }
    }
    
    private DashboardComponentDO createComponentDO(String code, String name, String type, String desc, int order) {
        DashboardComponentDO comp = new DashboardComponentDO();
        comp.setComponentCode(code);
        comp.setComponentName(name);
        comp.setComponentType(type);
        comp.setDescription(desc);
        comp.setSortOrder(order);
        return comp;
    }
}