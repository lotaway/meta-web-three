package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.ChecklistItem;
import com.metawebthree.mes.domain.repository.ChecklistItemRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ChecklistItemDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ChecklistItemMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class ChecklistItemRepositoryImpl implements ChecklistItemRepository {
    
    @Autowired
    private ChecklistItemMapper checklistItemMapper;
    
    @Override
    public Optional<ChecklistItem> findById(Long id) {
        ChecklistItemDO itemDO = checklistItemMapper.selectById(id);
        return Optional.ofNullable(itemDO).map(this::toEntity);
    }
    
    @Override
    public Optional<ChecklistItem> findByItemCode(String itemCode) {
        LambdaQueryWrapper<ChecklistItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ChecklistItemDO::getItemCode, itemCode);
        ChecklistItemDO itemDO = checklistItemMapper.selectOne(wrapper);
        return Optional.ofNullable(itemDO).map(this::toEntity);
    }
    
    @Override
    public List<ChecklistItem> findByItemCategory(String itemCategory) {
        LambdaQueryWrapper<ChecklistItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ChecklistItemDO::getItemCategory, itemCategory);
        List<ChecklistItemDO> doList = checklistItemMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ChecklistItem> findByStatus(ChecklistItem.ItemStatus status) {
        LambdaQueryWrapper<ChecklistItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ChecklistItemDO::getStatus, status.name());
        List<ChecklistItemDO> doList = checklistItemMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ChecklistItem> findAll() {
        List<ChecklistItemDO> doList = checklistItemMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public ChecklistItem save(ChecklistItem item) {
        ChecklistItemDO itemDO = toDO(item);
        if (item.getId() == null) {
            checklistItemMapper.insert(itemDO);
            item.setId(itemDO.getId());
        } else {
            checklistItemMapper.updateById(itemDO);
        }
        return item;
    }
    
    @Override
    public void update(ChecklistItem item) {
        checklistItemMapper.updateById(toDO(item));
    }
    
    @Override
    public void deleteById(Long id) {
        checklistItemMapper.deleteById(id);
    }
    
    private ChecklistItem toEntity(ChecklistItemDO doObj) {
        if (doObj == null) {
            return null;
        }
        ChecklistItem item = new ChecklistItem();
        item.setId(doObj.getId());
        item.setItemCode(doObj.getItemCode());
        item.setItemName(doObj.getItemName());
        item.setItemCategory(doObj.getItemCategory());
        item.setDataType(doObj.getDataType());
        item.setStandardValue(doObj.getStandardValue());
        item.setUpperLimit(doObj.getUpperLimit());
        item.setLowerLimit(doObj.getLowerLimit());
        item.setUnit(doObj.getUnit());
        item.setCheckMethod(doObj.getCheckMethod());
        item.setAbnormalJudgment(doObj.getAbnormalJudgment());
        item.setIsMandatory(doObj.getIsMandatory());
        item.setSortOrder(doObj.getSortOrder());
        item.setStatus(doObj.getStatus() != null ? ChecklistItem.ItemStatus.valueOf(doObj.getStatus()) : null);
        item.setRemark(doObj.getRemark());
        item.setCreatedAt(doObj.getCreatedAt());
        item.setUpdatedAt(doObj.getUpdatedAt());
        return item;
    }
    
    private ChecklistItemDO toDO(ChecklistItem item) {
        ChecklistItemDO doObj = new ChecklistItemDO();
        doObj.setId(item.getId());
        doObj.setItemCode(item.getItemCode());
        doObj.setItemName(item.getItemName());
        doObj.setItemCategory(item.getItemCategory());
        doObj.setDataType(item.getDataType());
        doObj.setStandardValue(item.getStandardValue());
        doObj.setUpperLimit(item.getUpperLimit());
        doObj.setLowerLimit(item.getLowerLimit());
        doObj.setUnit(item.getUnit());
        doObj.setCheckMethod(item.getCheckMethod());
        doObj.setAbnormalJudgment(item.getAbnormalJudgment());
        doObj.setIsMandatory(item.getIsMandatory());
        doObj.setSortOrder(item.getSortOrder());
        doObj.setStatus(item.getStatus() != null ? item.getStatus().name() : null);
        doObj.setRemark(item.getRemark());
        doObj.setCreatedAt(item.getCreatedAt());
        doObj.setUpdatedAt(item.getUpdatedAt());
        return doObj;
    }
}