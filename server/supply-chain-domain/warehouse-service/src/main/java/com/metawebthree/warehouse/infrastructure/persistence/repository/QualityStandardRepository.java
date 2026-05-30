package com.metawebthree.warehouse.infrastructure.persistence.repository;

import com.metawebthree.warehouse.domain.entity.QualityStandard;
import com.metawebthree.warehouse.infrastructure.persistence.converter.QualityStandardConverter;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.QualityStandardDO;
import com.metawebthree.warehouse.infrastructure.persistence.mapper.QualityStandardMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class QualityStandardRepository {
    
    @Autowired
    private QualityStandardMapper mapper;
    
    @Autowired
    private QualityStandardConverter converter;
    
    public QualityStandard save(QualityStandard entity) {
        QualityStandardDO dto = converter.toDO(entity);
        mapper.insert(dto);
        entity.setId(dto.getId());
        return entity;
    }
    
    public QualityStandard update(QualityStandard entity) {
        QualityStandardDO dto = converter.toDO(entity);
        mapper.updateById(dto);
        return entity;
    }
    
    public QualityStandard findById(Long id) {
        QualityStandardDO dto = mapper.selectById(id);
        return converter.toEntity(dto);
    }
    
    public QualityStandard findBySkuCode(String skuCode) {
        LambdaQueryWrapper<QualityStandardDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QualityStandardDO::getSkuCode, skuCode)
               .eq(QualityStandardDO::getDeleted, 0);
        QualityStandardDO dto = mapper.selectOne(wrapper);
        return converter.toEntity(dto);
    }
    
    public List<QualityStandard> findAll() {
        LambdaQueryWrapper<QualityStandardDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QualityStandardDO::getDeleted, 0)
               .orderByDesc(QualityStandardDO::getId);
        List<QualityStandardDO> dtoList = mapper.selectList(wrapper);
        return converter.toEntityList(dtoList);
    }
    
    public List<QualityStandard> findByActive(Boolean isActive) {
        LambdaQueryWrapper<QualityStandardDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QualityStandardDO::getDeleted, 0)
               .eq(QualityStandardDO::getIsActive, isActive ? 1 : 0)
               .orderByDesc(QualityStandardDO::getId);
        List<QualityStandardDO> dtoList = mapper.selectList(wrapper);
        return converter.toEntityList(dtoList);
    }
    
    private List<QualityStandard> toEntityList(List<QualityStandardDO> dtoList) {
        return dtoList.stream()
                .map(converter::toEntity)
                .collect(java.util.stream.Collectors.toList());
    }
}