package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.DataDictionary;
import com.metawebthree.mes.domain.repository.DataDictionaryRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.DataDictionaryDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.DataDictionaryItemDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.DataDictionaryItemMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.DataDictionaryMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class DataDictionaryRepositoryImpl implements DataDictionaryRepository {
    
    @Autowired
    private DataDictionaryMapper dataDictionaryMapper;
    
    @Autowired
    private DataDictionaryItemMapper dataDictionaryItemMapper;
    
    @Override
    public Optional<DataDictionary> findById(Long id) {
        DataDictionaryDO doObj = dataDictionaryMapper.selectById(id);
        if (doObj == null) {
            return Optional.empty();
        }
        return Optional.of(toEntity(doObj));
    }
    
    @Override
    public Optional<DataDictionary> findByDictCode(String dictCode) {
        LambdaQueryWrapper<DataDictionaryDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DataDictionaryDO::getDictCode, dictCode);
        DataDictionaryDO doObj = dataDictionaryMapper.selectOne(wrapper);
        if (doObj == null) {
            return Optional.empty();
        }
        return Optional.of(toEntity(doObj));
    }
    
    @Override
    public List<DataDictionary> findAllActive() {
        LambdaQueryWrapper<DataDictionaryDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DataDictionaryDO::getStatus, DataDictionary.DictStatus.ACTIVE.name())
               .orderByAsc(DataDictionaryDO::getSortOrder);
        List<DataDictionaryDO> doList = dataDictionaryMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public DataDictionary save(DataDictionary dictionary) {
        DataDictionaryDO doObj = toDO(dictionary);
        
        if (dictionary.getId() == null) {
            dataDictionaryMapper.insert(doObj);
            dictionary.setId(doObj.getId());
        } else {
            dataDictionaryMapper.updateById(doObj);
            LambdaQueryWrapper<DataDictionaryItemDO> deleteWrapper = new LambdaQueryWrapper<>();
            deleteWrapper.eq(DataDictionaryItemDO::getDictId, dictionary.getId());
            dataDictionaryItemMapper.delete(deleteWrapper);
        }
        
        if (dictionary.getItems() != null) {
            for (DataDictionary.DataDictionaryItem item : dictionary.getItems()) {
                DataDictionaryItemDO itemDO = toItemDO(item, dictionary.getId());
                dataDictionaryItemMapper.insert(itemDO);
                item.setId(itemDO.getId());
                item.setDictId(dictionary.getId());
            }
        }
        
        return dictionary;
    }
    
    @Override
    public void delete(Long id) {
        LambdaQueryWrapper<DataDictionaryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DataDictionaryItemDO::getDictId, id);
        dataDictionaryItemMapper.delete(wrapper);
        
        dataDictionaryMapper.deleteById(id);
    }
    
    @Override
    public boolean existsByDictCode(String dictCode) {
        LambdaQueryWrapper<DataDictionaryDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DataDictionaryDO::getDictCode, dictCode);
        return dataDictionaryMapper.selectCount(wrapper) > 0;
    }
    
    @Override
    public List<DataDictionary.DataDictionaryItem> findItemsByDictIdAndParentItemCode(Long dictId, String parentItemCode) {
        LambdaQueryWrapper<DataDictionaryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DataDictionaryItemDO::getDictId, dictId)
               .eq(DataDictionaryItemDO::getStatus, DataDictionary.DataDictionaryItem.ItemStatus.ACTIVE.name());
        
        if (parentItemCode == null) {
            wrapper.isNull(DataDictionaryItemDO::getParentItemCode);
        } else {
            wrapper.eq(DataDictionaryItemDO::getParentItemCode, parentItemCode);
        }
        
        wrapper.orderByAsc(DataDictionaryItemDO::getSortOrder);
        List<DataDictionaryItemDO> itemDOList = dataDictionaryItemMapper.selectList(wrapper);
        
        return itemDOList.stream().map(this::toItemEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<DataDictionary.DataDictionaryItem> findRootItemsByDictId(Long dictId) {
        return findItemsByDictIdAndParentItemCode(dictId, null);
    }
    
    private DataDictionary toEntity(DataDictionaryDO doObj) {
        if (doObj == null) {
            return null;
        }
        DataDictionary entity = new DataDictionary();
        entity.setId(doObj.getId());
        entity.setDictCode(doObj.getDictCode());
        entity.setDictName(doObj.getDictName());
        entity.setDescription(doObj.getDescription());
        entity.setStatus(DataDictionary.DictStatus.valueOf(doObj.getStatus()));
        entity.setSortOrder(doObj.getSortOrder());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        
        LambdaQueryWrapper<DataDictionaryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DataDictionaryItemDO::getDictId, doObj.getId())
               .orderByAsc(DataDictionaryItemDO::getSortOrder);
        List<DataDictionaryItemDO> itemDOList = dataDictionaryItemMapper.selectList(wrapper);
        
        if (itemDOList != null && !itemDOList.isEmpty()) {
            entity.setItems(itemDOList.stream().map(this::toItemEntity).collect(Collectors.toList()));
        } else {
            entity.setItems(new ArrayList<>());
        }
        
        return entity;
    }
    
    private DataDictionaryDO toDO(DataDictionary entity) {
        if (entity == null) {
            return null;
        }
        DataDictionaryDO doObj = new DataDictionaryDO();
        doObj.setId(entity.getId());
        doObj.setDictCode(entity.getDictCode());
        doObj.setDictName(entity.getDictName());
        doObj.setDescription(entity.getDescription());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : DataDictionary.DictStatus.ACTIVE.name());
        doObj.setSortOrder(entity.getSortOrder());
        return doObj;
    }
    
    private DataDictionary.DataDictionaryItem toItemEntity(DataDictionaryItemDO doObj) {
        if (doObj == null) {
            return null;
        }
        DataDictionary.DataDictionaryItem entity = new DataDictionary.DataDictionaryItem();
        entity.setId(doObj.getId());
        entity.setDictId(doObj.getDictId());
        entity.setItemCode(doObj.getItemCode());
        entity.setItemLabel(doObj.getItemLabel());
        entity.setParentItemCode(doObj.getParentItemCode());
        entity.setSortOrder(doObj.getSortOrder());
        entity.setStatus(DataDictionary.DataDictionaryItem.ItemStatus.valueOf(doObj.getStatus()));
        return entity;
    }
    
    private DataDictionaryItemDO toItemDO(DataDictionary.DataDictionaryItem entity, Long dictId) {
        if (entity == null) {
            return null;
        }
        DataDictionaryItemDO doObj = new DataDictionaryItemDO();
        doObj.setId(entity.getId());
        doObj.setDictId(dictId);
        doObj.setItemCode(entity.getItemCode());
        doObj.setItemLabel(entity.getItemLabel());
        doObj.setParentItemCode(entity.getParentItemCode());
        doObj.setSortOrder(entity.getSortOrder());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : DataDictionary.DataDictionaryItem.ItemStatus.ACTIVE.name());
        return doObj;
    }
}