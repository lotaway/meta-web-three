package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.ParameterGroupTemplate;
import com.metawebthree.mes.domain.repository.ParameterGroupTemplateRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ParameterGroupTemplateDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ParameterGroupTemplateMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class ParameterGroupTemplateRepositoryImpl implements ParameterGroupTemplateRepository {
    
    @Autowired
    private ParameterGroupTemplateMapper parameterGroupTemplateMapper;
    
    @Override
    public Optional<ParameterGroupTemplate> findById(Long id) {
        ParameterGroupTemplateDO doObj = parameterGroupTemplateMapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public Optional<ParameterGroupTemplate> findByTemplateCode(String templateCode) {
        LambdaQueryWrapper<ParameterGroupTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ParameterGroupTemplateDO::getTemplateCode, templateCode);
        ParameterGroupTemplateDO doObj = parameterGroupTemplateMapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public List<ParameterGroupTemplate> findByProductType(String productType) {
        LambdaQueryWrapper<ParameterGroupTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ParameterGroupTemplateDO::getProductType, productType)
                .orderByAsc(ParameterGroupTemplateDO::getDisplayOrder);
        List<ParameterGroupTemplateDO> doList = parameterGroupTemplateMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<ParameterGroupTemplate> findByStatus(ParameterGroupTemplate.TemplateStatus status) {
        LambdaQueryWrapper<ParameterGroupTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ParameterGroupTemplateDO::getStatus, status.name())
                .orderByAsc(ParameterGroupTemplateDO::getDisplayOrder);
        List<ParameterGroupTemplateDO> doList = parameterGroupTemplateMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<ParameterGroupTemplate> findByProductTypeAndStatus(String productType, ParameterGroupTemplate.TemplateStatus status) {
        LambdaQueryWrapper<ParameterGroupTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ParameterGroupTemplateDO::getProductType, productType)
                .eq(ParameterGroupTemplateDO::getStatus, status.name())
                .orderByAsc(ParameterGroupTemplateDO::getDisplayOrder);
        List<ParameterGroupTemplateDO> doList = parameterGroupTemplateMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<ParameterGroupTemplate> findAll() {
        LambdaQueryWrapper<ParameterGroupTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.orderByAsc(ParameterGroupTemplateDO::getDisplayOrder);
        List<ParameterGroupTemplateDO> doList = parameterGroupTemplateMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public boolean existsByTemplateCode(String templateCode) {
        LambdaQueryWrapper<ParameterGroupTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ParameterGroupTemplateDO::getTemplateCode, templateCode);
        return parameterGroupTemplateMapper.selectCount(wrapper) > 0;
    }
    
    @Override
    public boolean existsByProductType(String productType) {
        LambdaQueryWrapper<ParameterGroupTemplateDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ParameterGroupTemplateDO::getProductType, productType);
        return parameterGroupTemplateMapper.selectCount(wrapper) > 0;
    }
    
    @Override
    public long count() {
        return parameterGroupTemplateMapper.selectCount(null);
    }
    
    @Override
    public ParameterGroupTemplate save(ParameterGroupTemplate template) {
        ParameterGroupTemplateDO doObj = toDO(template);
        if (template.getId() == null) {
            parameterGroupTemplateMapper.insert(doObj);
            template.setId(doObj.getId());
        } else {
            parameterGroupTemplateMapper.updateById(doObj);
        }
        return template;
    }
    
    @Override
    public void deleteById(Long id) {
        parameterGroupTemplateMapper.deleteById(id);
    }
    
    @Override
    public boolean existsById(Long id) {
        return parameterGroupTemplateMapper.selectById(id) != null;
    }
    
    private ParameterGroupTemplate toEntity(ParameterGroupTemplateDO doObj) {
        if (doObj == null) {
            return null;
        }
        ParameterGroupTemplate entity = new ParameterGroupTemplate();
        entity.setId(doObj.getId());
        entity.setTemplateCode(doObj.getTemplateCode());
        entity.setTemplateName(doObj.getTemplateName());
        entity.setProductType(doObj.getProductType());
        entity.setDescription(doObj.getDescription());
        entity.setStatus(doObj.getStatus() != null ? ParameterGroupTemplate.TemplateStatus.valueOf(doObj.getStatus()) : null);
        entity.setDisplayOrder(doObj.getDisplayOrder());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedBy(doObj.getUpdatedBy());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        
        // 解析参数ID列表
        if (doObj.getParameterIds() != null && !doObj.getParameterIds().isEmpty()) {
            List<Long> ids = Arrays.stream(doObj.getParameterIds().split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .map(Long::parseLong)
                    .collect(Collectors.toList());
            entity.setParameterIds(ids);
        }
        
        // 解析参数编码列表
        if (doObj.getParameterCodes() != null && !doObj.getParameterCodes().isEmpty()) {
            List<String> codes = Arrays.stream(doObj.getParameterCodes().split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .collect(Collectors.toList());
            entity.setParameterCodes(codes);
        }
        
        return entity;
    }
    
    private ParameterGroupTemplateDO toDO(ParameterGroupTemplate entity) {
        if (entity == null) {
            return null;
        }
        ParameterGroupTemplateDO doObj = new ParameterGroupTemplateDO();
        doObj.setId(entity.getId());
        doObj.setTemplateCode(entity.getTemplateCode());
        doObj.setTemplateName(entity.getTemplateName());
        doObj.setProductType(entity.getProductType());
        doObj.setDescription(entity.getDescription());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setDisplayOrder(entity.getDisplayOrder());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setUpdatedBy(entity.getUpdatedBy());
        
        // 序列化参数ID列表
        if (entity.getParameterIds() != null && !entity.getParameterIds().isEmpty()) {
            doObj.setParameterIds(entity.getParameterIds().stream()
                    .map(String::valueOf)
                    .collect(Collectors.joining(",")));
        }
        
        // 序列化参数编码列表
        if (entity.getParameterCodes() != null && !entity.getParameterCodes().isEmpty()) {
            doObj.setParameterCodes(String.join(",", entity.getParameterCodes()));
        }
        
        return doObj;
    }
}