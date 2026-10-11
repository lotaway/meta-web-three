package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.EquipmentChecklistTemplate;
import com.metawebthree.mes.domain.repository.EquipmentChecklistTemplateRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentChecklistTemplateDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.EquipmentChecklistTemplateMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class EquipmentChecklistTemplateRepositoryImpl implements EquipmentChecklistTemplateRepository {
    
    @Autowired
    private EquipmentChecklistTemplateMapper templateMapper;
    
    @Override
    public Optional<EquipmentChecklistTemplate> findById(Long id) {
        EquipmentChecklistTemplateDO templateDO = templateMapper.selectById(id);
        return Optional.ofNullable(templateDO).map(this::toEntity);
    }
    
    @Override
    public Optional<EquipmentChecklistTemplate> findByTemplateCode(String templateCode) {
        LambdaQueryWrapper<EquipmentChecklistTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistTemplateDO::getTemplateCode, templateCode);
        EquipmentChecklistTemplateDO templateDO = templateMapper.selectOne(wrapper);
        return Optional.ofNullable(templateDO).map(this::toEntity);
    }
    
    @Override
    public List<EquipmentChecklistTemplate> findByEquipmentTypeCode(String equipmentTypeCode) {
        LambdaQueryWrapper<EquipmentChecklistTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistTemplateDO::getEquipmentTypeCode, equipmentTypeCode);
        List<EquipmentChecklistTemplateDO> doList = templateMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<EquipmentChecklistTemplate> findByStatus(EquipmentChecklistTemplate.TemplateStatus status) {
        LambdaQueryWrapper<EquipmentChecklistTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistTemplateDO::getStatus, status.name());
        List<EquipmentChecklistTemplateDO> doList = templateMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<EquipmentChecklistTemplate> findAll() {
        List<EquipmentChecklistTemplateDO> doList = templateMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public EquipmentChecklistTemplate save(EquipmentChecklistTemplate template) {
        EquipmentChecklistTemplateDO templateDO = toDO(template);
        if (template.getId() == null) {
            templateMapper.insert(templateDO);
            template.setId(templateDO.getId());
        } else {
            templateMapper.updateById(templateDO);
        }
        return template;
    }
    
    @Override
    public void update(EquipmentChecklistTemplate template) {
        templateMapper.updateById(toDO(template));
    }
    
    @Override
    public void deleteById(Long id) {
        templateMapper.deleteById(id);
    }
    
    private EquipmentChecklistTemplate toEntity(EquipmentChecklistTemplateDO doObj) {
        if (doObj == null) {
            return null;
        }
        EquipmentChecklistTemplate template = new EquipmentChecklistTemplate();
        template.setId(doObj.getId());
        template.setTemplateCode(doObj.getTemplateCode());
        template.setTemplateName(doObj.getTemplateName());
        template.setEquipmentTypeCode(doObj.getEquipmentTypeCode());
        template.setCheckPeriodType(doObj.getCheckPeriodType());
        template.setCheckPeriodValue(doObj.getCheckPeriodValue());
        template.setCheckPeriodUnit(doObj.getCheckPeriodUnit());
        template.setRunningHoursThreshold(doObj.getRunningHoursThreshold());
        template.setAlertBeforeHours(doObj.getAlertBeforeHours());
        template.setStatus(doObj.getStatus() != null ? EquipmentChecklistTemplate.TemplateStatus.valueOf(doObj.getStatus()) : null);
        template.setVersion(doObj.getVersion());
        template.setRemark(doObj.getRemark());
        template.setCreatedAt(doObj.getCreatedAt());
        template.setUpdatedAt(doObj.getUpdatedAt());
        return template;
    }
    
    private EquipmentChecklistTemplateDO toDO(EquipmentChecklistTemplate template) {
        EquipmentChecklistTemplateDO doObj = new EquipmentChecklistTemplateDO();
        doObj.setId(template.getId());
        doObj.setTemplateCode(template.getTemplateCode());
        doObj.setTemplateName(template.getTemplateName());
        doObj.setEquipmentTypeCode(template.getEquipmentTypeCode());
        doObj.setCheckPeriodType(template.getCheckPeriodType());
        doObj.setCheckPeriodValue(template.getCheckPeriodValue());
        doObj.setCheckPeriodUnit(template.getCheckPeriodUnit());
        doObj.setRunningHoursThreshold(template.getRunningHoursThreshold());
        doObj.setAlertBeforeHours(template.getAlertBeforeHours());
        doObj.setStatus(template.getStatus() != null ? template.getStatus().name() : null);
        doObj.setVersion(template.getVersion());
        doObj.setRemark(template.getRemark());
        doObj.setCreatedAt(template.getCreatedAt());
        doObj.setUpdatedAt(template.getUpdatedAt());
        return doObj;
    }
}