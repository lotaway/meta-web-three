package com.metawebthree.mes.infrastructure.persistence.repository.trace;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.TraceDataScope;
import com.metawebthree.mes.domain.entity.TraceDataScope.DataScopeType;
import com.metawebthree.mes.domain.entity.TraceDataScope.ScopeItem;
import com.metawebthree.mes.domain.repository.trace.TraceDataScopeRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.trace.TraceDataScopeDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.trace.TraceDataScopeMapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class TraceDataScopeRepositoryImpl implements TraceDataScopeRepository {

    private final TraceDataScopeMapper mapper;
    private final ObjectMapper objectMapper;

    public TraceDataScopeRepositoryImpl(TraceDataScopeMapper mapper, ObjectMapper objectMapper) {
        this.mapper = mapper;
        this.objectMapper = objectMapper;
    }

    @Override
    public Optional<TraceDataScope> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id)).map(this::toEntity);
    }

    @Override
    public Optional<TraceDataScope> findByScopeCode(String scopeCode) {
        LambdaQueryWrapper<TraceDataScopeDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceDataScopeDO::getScopeCode, scopeCode);
        return Optional.ofNullable(mapper.selectOne(w)).map(this::toEntity);
    }

    @Override
    public List<TraceDataScope> findByScopeType(DataScopeType scopeType) {
        LambdaQueryWrapper<TraceDataScopeDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceDataScopeDO::getScopeType, scopeType.name());
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TraceDataScope> findByIsDefault(Boolean isDefault) {
        LambdaQueryWrapper<TraceDataScopeDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceDataScopeDO::getIsDefault, isDefault);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TraceDataScope> findAll() {
        return mapper.selectList(null).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public TraceDataScope save(TraceDataScope entity) {
        TraceDataScopeDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            mapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return entity;
    }

    @Override
    public void update(TraceDataScope entity) {
        if (entity.getId() != null) {
            mapper.updateById(toDO(entity));
        }
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    @SuppressWarnings("unchecked")
    private TraceDataScope toEntity(TraceDataScopeDO doObj) {
        if (doObj == null) return null;
        TraceDataScope entity = new TraceDataScope();
        entity.setId(doObj.getId());
        entity.setScopeCode(doObj.getScopeCode());
        entity.setScopeName(doObj.getScopeName());
        entity.setScopeType(doObj.getScopeType() != null ? DataScopeType.valueOf(doObj.getScopeType()) : null);
        entity.setIsDefault(doObj.getIsDefault() != null ? doObj.getIsDefault() : false);
        List<Map<String, Object>> rawItems = doObj.getItems();
        if (rawItems != null) {
            List<ScopeItem> items = new ArrayList<>();
            for (Map<String, Object> m : rawItems) {
                ScopeItem item = new ScopeItem();
                item.setItemCode((String) m.get("itemCode"));
                item.setItemName((String) m.get("itemName"));
                if (m.get("dataType") != null)
                    item.setDataType(DataScopeType.valueOf((String) m.get("dataType")));
                item.setIsRequired((Boolean) m.getOrDefault("isRequired", false));
                item.setRetentionDays((Integer) m.get("retentionDays"));
                item.setDescription((String) m.get("description"));
                items.add(item);
            }
            entity.setItems(items);
        }
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private TraceDataScopeDO toDO(TraceDataScope entity) {
        if (entity == null) return null;
        TraceDataScopeDO doObj = new TraceDataScopeDO();
        doObj.setId(entity.getId());
        doObj.setScopeCode(entity.getScopeCode());
        doObj.setScopeName(entity.getScopeName());
        doObj.setScopeType(entity.getScopeType() != null ? entity.getScopeType().name() : null);
        doObj.setIsDefault(entity.getIsDefault());
        if (entity.getItems() != null) {
            List<Map<String, Object>> itemMaps = new ArrayList<>();
            for (ScopeItem item : entity.getItems()) {
                try {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> m = objectMapper.convertValue(item, Map.class);
                    itemMaps.add(m);
                } catch (Exception e) {
                    itemMaps.add(Collections.emptyMap());
                }
            }
            doObj.setItems(itemMaps);
        }
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
