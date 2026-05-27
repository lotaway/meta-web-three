package com.metawebthree.mes.application.command;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessFlowTemplateDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessFlowTemplateVersionDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProcessFlowTemplateMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProcessFlowTemplateVersionMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ProcessFlowCommandService {
    
    private final ProcessFlowTemplateMapper templateMapper;
    private final ProcessFlowTemplateVersionMapper versionMapper;
    
    @Transactional
    public ProcessFlowTemplateDO createTemplate(ProcessFlowTemplateDO template) {
        template.setVersion(1);
        template.setStatus("DRAFT");
        template.setDeleted(false);
        template.setCreatedAt(LocalDateTime.now());
        templateMapper.insert(template);
        return template;
    }
    
    @Transactional
    public ProcessFlowTemplateDO updateTemplate(ProcessFlowTemplateDO template) {
        ProcessFlowTemplateDO existing = templateMapper.selectById(template.getId());
        if (existing != null) {
            saveVersionHistory(existing, "Update template");
        }
        template.setUpdatedAt(LocalDateTime.now());
        templateMapper.updateById(template);
        return template;
    }
    
    @Transactional
    public void saveVersionHistory(ProcessFlowTemplateDO template, String changeDescription) {
        ProcessFlowTemplateVersionDO version = new ProcessFlowTemplateVersionDO();
        version.setTemplateId(template.getId());
        version.setVersion(template.getVersion());
        version.setTemplateCode(template.getTemplateCode());
        version.setTemplateName(template.getTemplateName());
        version.setDescription(template.getDescription());
        version.setFlowData(template.getFlowData());
        version.setStatus(template.getStatus());
        version.setChangeDescription(changeDescription);
        version.setIsCurrentVersion(true);
        version.setCreatedBy(template.getUpdatedBy());
        version.setCreatedAt(LocalDateTime.now());
        version.setDeleted(false);
        versionMapper.insert(version);
        
        LambdaQueryWrapper<ProcessFlowTemplateVersionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessFlowTemplateVersionDO::getTemplateId, template.getId())
               .ne(ProcessFlowTemplateVersionDO::getId, version.getId());
        versionMapper.delete(wrapper);
    }
    
    public List<ProcessFlowTemplateVersionDO> getVersionHistory(Long templateId) {
        LambdaQueryWrapper<ProcessFlowTemplateVersionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessFlowTemplateVersionDO::getTemplateId, templateId)
               .eq(ProcessFlowTemplateVersionDO::getDeleted, false)
               .orderByDesc(ProcessFlowTemplateVersionDO::getVersion);
        return versionMapper.selectList(wrapper);
    }
    
    @Transactional
    public ProcessFlowTemplateDO rollbackToVersion(Long templateId, Integer targetVersion, Long userId) {
        LambdaQueryWrapper<ProcessFlowTemplateVersionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessFlowTemplateVersionDO::getTemplateId, templateId)
               .eq(ProcessFlowTemplateVersionDO::getVersion, targetVersion);
        ProcessFlowTemplateVersionDO targetVersionData = versionMapper.selectOne(wrapper);
        
        if (targetVersionData == null) {
            throw new RuntimeException("Version not found: " + targetVersion);
        }
        
        ProcessFlowTemplateDO currentTemplate = templateMapper.selectById(templateId);
        if (currentTemplate != null) {
            saveVersionHistory(currentTemplate, "Rollback to version " + targetVersion);
            
            currentTemplate.setTemplateName(targetVersionData.getTemplateName());
            currentTemplate.setDescription(targetVersionData.getDescription());
            currentTemplate.setFlowData(targetVersionData.getFlowData());
            currentTemplate.setVersion(currentTemplate.getVersion() + 1);
            currentTemplate.setUpdatedBy(userId);
            currentTemplate.setUpdatedAt(LocalDateTime.now());
            templateMapper.updateById(currentTemplate);
        }
        return currentTemplate;
    }
    
    @Transactional
    public void publishTemplate(Long templateId, Long userId) {
        ProcessFlowTemplateDO template = templateMapper.selectById(templateId);
        if (template != null) {
            template.setStatus("PUBLISHED");
            template.setUpdatedBy(userId);
            template.setUpdatedAt(LocalDateTime.now());
            templateMapper.updateById(template);
        }
    }
    
    @Transactional
    public void archiveTemplate(Long templateId, Long userId) {
        ProcessFlowTemplateDO template = templateMapper.selectById(templateId);
        if (template != null) {
            template.setStatus("ARCHIVED");
            template.setUpdatedBy(userId);
            template.setUpdatedAt(LocalDateTime.now());
            templateMapper.updateById(template);
        }
    }
    
    @Transactional
    public void deleteTemplate(Long templateId) {
        ProcessFlowTemplateDO template = templateMapper.selectById(templateId);
        if (template != null) {
            template.setDeleted(true);
            template.setUpdatedAt(LocalDateTime.now());
            templateMapper.updateById(template);
        }
    }
    
    public List<ProcessFlowTemplateDO> listTemplates(String status) {
        LambdaQueryWrapper<ProcessFlowTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessFlowTemplateDO::getDeleted, false);
        if (status != null && !status.isEmpty()) {
            wrapper.eq(ProcessFlowTemplateDO::getStatus, status);
        }
        wrapper.orderByDesc(ProcessFlowTemplateDO::getCreatedAt);
        return templateMapper.selectList(wrapper);
    }
    
    public ProcessFlowTemplateDO getTemplate(Long templateId) {
        return templateMapper.selectById(templateId);
    }
}