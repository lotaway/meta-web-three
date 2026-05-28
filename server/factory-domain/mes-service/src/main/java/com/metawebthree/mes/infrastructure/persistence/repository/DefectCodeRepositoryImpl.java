package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.DefectCode;
import com.metawebthree.mes.domain.repository.DefectCodeRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.DefectCodeDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.DefectCodeMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class DefectCodeRepositoryImpl implements DefectCodeRepository {
    
    @Autowired
    private DefectCodeMapper defectCodeMapper;
    
    @Override
    public Optional<DefectCode> findById(Long id) {
        DefectCodeDO defectCodeDO = defectCodeMapper.selectById(id);
        return Optional.ofNullable(defectCodeDO).map(this::toEntity);
    }
    
    @Override
    public Optional<DefectCode> findByDefectCode(String defectCode) {
        LambdaQueryWrapper<DefectCodeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DefectCodeDO::getDefectCode, defectCode);
        DefectCodeDO defectCodeDO = defectCodeMapper.selectOne(wrapper);
        return Optional.ofNullable(defectCodeDO).map(this::toEntity);
    }
    
    @Override
    public List<DefectCode> findAll() {
        List<DefectCodeDO> doList = defectCodeMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<DefectCode> findByCategory(DefectCode.DefectCategory category) {
        LambdaQueryWrapper<DefectCodeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DefectCodeDO::getCategory, category.name());
        List<DefectCodeDO> doList = defectCodeMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<DefectCode> findBySeverity(DefectCode.DefectSeverity severity) {
        LambdaQueryWrapper<DefectCodeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DefectCodeDO::getSeverity, severity.name());
        List<DefectCodeDO> doList = defectCodeMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<DefectCode> findByIsEnabled(Boolean isEnabled) {
        LambdaQueryWrapper<DefectCodeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DefectCodeDO::getIsEnabled, isEnabled);
        List<DefectCodeDO> doList = defectCodeMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public DefectCode save(DefectCode defectCode) {
        DefectCodeDO defectCodeDO = toDO(defectCode);
        if (defectCode.getId() == null) {
            defectCodeMapper.insert(defectCodeDO);
            defectCode.setId(defectCodeDO.getId());
        } else {
            defectCodeMapper.updateById(defectCodeDO);
        }
        return defectCode;
    }
    
    @Override
    public void update(DefectCode defectCode) {
        if (defectCode.getId() != null) {
            DefectCodeDO defectCodeDO = toDO(defectCode);
            defectCodeMapper.updateById(defectCodeDO);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        defectCodeMapper.deleteById(id);
    }
    
    @Override
    public Boolean existsByDefectCode(String defectCode) {
        LambdaQueryWrapper<DefectCodeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DefectCodeDO::getDefectCode, defectCode);
        return defectCodeMapper.selectCount(wrapper) > 0;
    }
    
    private DefectCode toEntity(DefectCodeDO doObj) {
        if (doObj == null) {
            return null;
        }
        DefectCode entity = new DefectCode();
        entity.setId(doObj.getId());
        entity.setDefectCode(doObj.getDefectCode());
        entity.setDefectName(doObj.getDefectName());
        if (doObj.getCategory() != null) {
            entity.setCategory(DefectCode.DefectCategory.valueOf(doObj.getCategory()));
        }
        if (doObj.getSeverity() != null) {
            entity.setSeverity(DefectCode.DefectSeverity.valueOf(doObj.getSeverity()));
        }
        entity.setIsCritical(doObj.getIsCritical());
        entity.setDescription(doObj.getDescription());
        entity.setDispositionGuide(doObj.getDispositionGuide());
        entity.setIsEnabled(doObj.getIsEnabled());
        entity.setSortOrder(doObj.getSortOrder());
        return entity;
    }
    
    private DefectCodeDO toDO(DefectCode entity) {
        if (entity == null) {
            return null;
        }
        DefectCodeDO doObj = new DefectCodeDO();
        doObj.setId(entity.getId());
        doObj.setDefectCode(entity.getDefectCode());
        doObj.setDefectName(entity.getDefectName());
        doObj.setCategory(entity.getCategory() != null ? entity.getCategory().name() : null);
        doObj.setSeverity(entity.getSeverity() != null ? entity.getSeverity().name() : null);
        doObj.setIsCritical(entity.getIsCritical());
        doObj.setDescription(entity.getDescription());
        doObj.setDispositionGuide(entity.getDispositionGuide());
        doObj.setIsEnabled(entity.getIsEnabled());
        doObj.setSortOrder(entity.getSortOrder());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}