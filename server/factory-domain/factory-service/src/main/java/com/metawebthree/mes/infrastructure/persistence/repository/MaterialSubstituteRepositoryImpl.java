package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.MaterialSubstitute;
import com.metawebthree.mes.domain.entity.MaterialSubstitute.SubstituteItem;
import com.metawebthree.mes.domain.repository.MaterialSubstituteRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialSubstituteDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialSubstituteItemDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.MaterialSubstituteMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.MaterialSubstituteItemMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class MaterialSubstituteRepositoryImpl implements MaterialSubstituteRepository {

    private final MaterialSubstituteMapper mapper;
    private final MaterialSubstituteItemMapper itemMapper;

    public MaterialSubstituteRepositoryImpl(MaterialSubstituteMapper mapper,
                                            MaterialSubstituteItemMapper itemMapper) {
        this.mapper = mapper;
        this.itemMapper = itemMapper;
    }

    @Override
    public Optional<MaterialSubstitute> findById(Long id) {
        MaterialSubstituteDO obj = mapper.selectById(id);
        if (obj == null) return Optional.empty();
        
        MaterialSubstitute entity = toEntity(obj);
        entity.setSubstitutes(loadItemsByGroupId(id));
        return Optional.of(entity);
    }

    @Override
    public Optional<MaterialSubstitute> findByProductCodeAndMainMaterialCode(String productCode, String mainMaterialCode) {
        MaterialSubstituteDO obj = mapper.findByProductCodeAndMainMaterialCode(productCode, mainMaterialCode);
        if (obj == null) return Optional.empty();
        
        MaterialSubstitute entity = toEntity(obj);
        entity.setSubstitutes(loadItemsByGroupId(obj.getId()));
        return Optional.of(entity);
    }

    @Override
    public List<MaterialSubstitute> findByProductCode(String productCode) {
        List<MaterialSubstituteDO> list = mapper.findByProductCode(productCode);
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<MaterialSubstitute> findActiveByProductCode(String productCode) {
        List<MaterialSubstituteDO> list = mapper.findActiveByProductCode(productCode);
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public MaterialSubstitute save(MaterialSubstitute substitute) {
        MaterialSubstituteDO DO = toDO(substitute);
        if (substitute.getId() == null) {
            mapper.insert(DO);
            substitute.setId(DO.getId());
        } else {
            mapper.updateById(DO);
        }
        
        // 保存子项
        if (substitute.getSubstitutes() != null && !substitute.getSubstitutes().isEmpty()) {
            for (SubstituteItem item : substitute.getSubstitutes()) {
                item.setSubstituteGroupId(substitute.getId());
                saveSubstituteItem(item);
            }
        }
        return substitute;
    }

    @Override
    public void update(MaterialSubstitute substitute) {
        mapper.updateById(toDO(substitute));
    }

    @Override
    public void deleteById(Long id) {
        // 先删除子项
        itemMapper.deleteByGroupId(id);
        mapper.deleteById(id);
    }

    private List<SubstituteItem> loadItemsByGroupId(Long groupId) {
        List<MaterialSubstituteItemDO> items = itemMapper.findByGroupId(groupId);
        return items.stream().map(this::toItemEntity).collect(Collectors.toList());
    }

    private void saveSubstituteItem(SubstituteItem item) {
        MaterialSubstituteItemDO DO = toItemDO(item);
        if (item.getId() == null) {
            itemMapper.insert(DO);
            item.setId(DO.getId());
        } else {
            itemMapper.updateById(DO);
        }
    }

    // ========== Entity <-> DO 转换方法 ==========

    private MaterialSubstitute toEntity(MaterialSubstituteDO obj) {
        if (obj == null) return null;
        MaterialSubstitute entity = new MaterialSubstitute();
        entity.setId(obj.getId());
        entity.setProductCode(obj.getProductCode());
        entity.setMainMaterialCode(obj.getMainMaterialCode());
        entity.setMainMaterialName(obj.getMainMaterialName());
        entity.setStatus(obj.getStatus());
        entity.setCreatedBy(obj.getCreatedBy());
        entity.setUpdatedBy(obj.getUpdatedBy());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }

    private MaterialSubstituteDO toDO(MaterialSubstitute entity) {
        if (entity == null) return null;
        MaterialSubstituteDO DO = new MaterialSubstituteDO();
        DO.setId(entity.getId());
        DO.setProductCode(entity.getProductCode());
        DO.setMainMaterialCode(entity.getMainMaterialCode());
        DO.setMainMaterialName(entity.getMainMaterialName());
        DO.setStatus(entity.getStatus());
        DO.setCreatedBy(entity.getCreatedBy());
        DO.setUpdatedBy(entity.getUpdatedBy());
        DO.setCreatedAt(entity.getCreatedAt());
        DO.setUpdatedAt(entity.getUpdatedAt());
        return DO;
    }

    private SubstituteItem toItemEntity(MaterialSubstituteItemDO obj) {
        if (obj == null) return null;
        SubstituteItem entity = new SubstituteItem();
        entity.setId(obj.getId());
        entity.setSubstituteGroupId(obj.getSubstituteGroupId());
        entity.setMaterialCode(obj.getMaterialCode());
        entity.setMaterialName(obj.getMaterialName());
        entity.setMaterialSpec(obj.getMaterialSpec());
        entity.setUnitCode(obj.getUnitCode());
        entity.setUnitName(obj.getUnitName());
        entity.setPriority(obj.getPriority());
        entity.setConversionRate(obj.getConversionRate());
        entity.setConversionUnit(obj.getConversionUnit());
        entity.setReason(obj.getReason());
        entity.setEffectiveDate(obj.getEffectiveDate());
        entity.setExpiryDate(obj.getExpiryDate());
        entity.setStatus(obj.getStatus());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }

    private MaterialSubstituteItemDO toItemDO(SubstituteItem entity) {
        if (entity == null) return null;
        MaterialSubstituteItemDO DO = new MaterialSubstituteItemDO();
        DO.setId(entity.getId());
        DO.setSubstituteGroupId(entity.getSubstituteGroupId());
        DO.setMaterialCode(entity.getMaterialCode());
        DO.setMaterialName(entity.getMaterialName());
        DO.setMaterialSpec(entity.getMaterialSpec());
        DO.setUnitCode(entity.getUnitCode());
        DO.setUnitName(entity.getUnitName());
        DO.setPriority(entity.getPriority());
        DO.setConversionRate(entity.getConversionRate());
        DO.setConversionUnit(entity.getConversionUnit());
        DO.setReason(entity.getReason());
        DO.setEffectiveDate(entity.getEffectiveDate());
        DO.setExpiryDate(entity.getExpiryDate());
        DO.setStatus(entity.getStatus());
        DO.setCreatedAt(entity.getCreatedAt());
        DO.setUpdatedAt(entity.getUpdatedAt());
        return DO;
    }
}