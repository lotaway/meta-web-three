package com.metawebthree.mes.infrastructure.persistence.repository.trace;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.TraceModel;
import com.metawebthree.mes.domain.entity.TraceModel.TraceRelationConfig;
import com.metawebthree.mes.domain.entity.TraceModel.TraceRelationConfig.TraceLevel;
import com.metawebthree.mes.domain.entity.TraceModel.TraceType;
import com.metawebthree.mes.domain.repository.trace.TraceModelRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.trace.TraceModelDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.trace.TraceModelMapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class TraceModelRepositoryImpl implements TraceModelRepository {

    private final TraceModelMapper mapper;
    private final ObjectMapper objectMapper;

    public TraceModelRepositoryImpl(TraceModelMapper mapper, ObjectMapper objectMapper) {
        this.mapper = mapper;
        this.objectMapper = objectMapper;
    }

    @Override
    public Optional<TraceModel> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id)).map(this::toEntity);
    }

    @Override
    public Optional<TraceModel> findByModelCode(String modelCode) {
        LambdaQueryWrapper<TraceModelDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceModelDO::getModelCode, modelCode);
        return Optional.ofNullable(mapper.selectOne(w)).map(this::toEntity);
    }

    @Override
    public List<TraceModel> findByProductType(String productType) {
        LambdaQueryWrapper<TraceModelDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceModelDO::getProductType, productType);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TraceModel> findByIsEnabled(Boolean isEnabled) {
        LambdaQueryWrapper<TraceModelDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceModelDO::getIsEnabled, isEnabled);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TraceModel> findAll() {
        return mapper.selectList(null).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public TraceModel save(TraceModel entity) {
        TraceModelDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            mapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return entity;
    }

    @Override
    public void update(TraceModel entity) {
        if (entity.getId() != null) {
            mapper.updateById(toDO(entity));
        }
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private TraceModel toEntity(TraceModelDO doObj) {
        if (doObj == null) return null;
        TraceModel entity = new TraceModel();
        entity.setId(doObj.getId());
        entity.setModelCode(doObj.getModelCode());
        entity.setModelName(doObj.getModelName());
        entity.setProductType(doObj.getProductType());
        entity.setRelationConfig(parseConfig(doObj.getRelationConfig()));
        entity.setIsEnabled(doObj.getIsEnabled() != null ? doObj.getIsEnabled() : true);
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private TraceModelDO toDO(TraceModel entity) {
        if (entity == null) return null;
        TraceModelDO doObj = new TraceModelDO();
        doObj.setId(entity.getId());
        doObj.setModelCode(entity.getModelCode());
        doObj.setModelName(entity.getModelName());
        doObj.setProductType(entity.getProductType());
        doObj.setRelationConfig(serializeConfig(entity.getRelationConfig()));
        doObj.setIsEnabled(entity.getIsEnabled());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }

    @SuppressWarnings("unchecked")
    private TraceRelationConfig parseConfig(Map<String, Object> raw) {
        if (raw == null) return null;
        TraceRelationConfig cfg = new TraceRelationConfig();
        cfg.setEnableBatchTrace((Boolean) raw.getOrDefault("enableBatchTrace", false));
        cfg.setEnableSnTrace((Boolean) raw.getOrDefault("enableSnTrace", false));
        cfg.setEnableMaterialTrace((Boolean) raw.getOrDefault("enableMaterialTrace", false));
        cfg.setEnableProcessTrace((Boolean) raw.getOrDefault("enableProcessTrace", false));
        cfg.setEnableQualityTrace((Boolean) raw.getOrDefault("enableQualityTrace", false));
        cfg.setEnableEquipmentTrace((Boolean) raw.getOrDefault("enableEquipmentTrace", false));
        List<Map<String, Object>> levels = (List<Map<String, Object>>) raw.get("traceLevels");
        if (levels != null) {
            List<TraceLevel> tl = new ArrayList<>();
            for (Map<String, Object> lm : levels) {
                TraceLevel l = new TraceLevel();
                l.setLevelCode((String) lm.get("levelCode"));
                l.setLevelName((String) lm.get("levelName"));
                if (lm.get("traceType") != null)
                    l.setTraceType(TraceType.valueOf((String) lm.get("traceType")));
                l.setParentLevelCode((String) lm.get("parentLevelCode"));
                l.setIsRequired((Boolean) lm.getOrDefault("isRequired", false));
                tl.add(l);
            }
            cfg.setTraceLevels(tl);
        }
        return cfg;
    }

    private Map<String, Object> serializeConfig(TraceRelationConfig cfg) {
        if (cfg == null) return Collections.emptyMap();
        try {
            return objectMapper.convertValue(cfg, Map.class);
        } catch (Exception e) {
            return Collections.emptyMap();
        }
    }
}
