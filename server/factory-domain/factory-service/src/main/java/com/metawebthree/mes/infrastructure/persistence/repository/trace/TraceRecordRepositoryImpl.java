package com.metawebthree.mes.infrastructure.persistence.repository.trace;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.TraceRecord;
import com.metawebthree.mes.domain.entity.TraceRecord.TraceRelation;
import com.metawebthree.mes.domain.entity.TraceRecord.TraceSource;
import com.metawebthree.mes.domain.entity.TraceRecord.TraceType;
import com.metawebthree.mes.domain.repository.trace.TraceRecordRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.trace.TraceRecordDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.trace.TraceRecordMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class TraceRecordRepositoryImpl implements TraceRecordRepository {

    private final TraceRecordMapper mapper;
    private final ObjectMapper objectMapper;

    public TraceRecordRepositoryImpl(TraceRecordMapper mapper, ObjectMapper objectMapper) {
        this.mapper = mapper;
        this.objectMapper = objectMapper;
    }

    @Override
    public Optional<TraceRecord> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id)).map(this::toEntity);
    }

    @Override
    public Optional<TraceRecord> findByTraceCode(String traceCode) {
        LambdaQueryWrapper<TraceRecordDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceRecordDO::getTraceCode, traceCode);
        return Optional.ofNullable(mapper.selectOne(w)).map(this::toEntity);
    }

    @Override
    public List<TraceRecord> findByBatchNo(String batchNo) {
        LambdaQueryWrapper<TraceRecordDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceRecordDO::getBatchNo, batchNo);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TraceRecord> findByProductCode(String productCode) {
        LambdaQueryWrapper<TraceRecordDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceRecordDO::getProductCode, productCode);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TraceRecord> findBySn(String sn) {
        LambdaQueryWrapper<TraceRecordDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceRecordDO::getSn, sn);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TraceRecord> findByWorkOrderNo(String workOrderNo) {
        LambdaQueryWrapper<TraceRecordDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceRecordDO::getWorkOrderNo, workOrderNo);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TraceRecord> findBySource(String source) {
        LambdaQueryWrapper<TraceRecordDO> w = new LambdaQueryWrapper<>();
        w.eq(TraceRecordDO::getSource, source);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TraceRecord> findAll() {
        return mapper.selectList(null).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public TraceRecord save(TraceRecord entity) {
        TraceRecordDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            mapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return entity;
    }

    @Override
    public void update(TraceRecord entity) {
        if (entity.getId() != null) {
            mapper.updateById(toDO(entity));
        }
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private TraceRecord toEntity(TraceRecordDO doObj) {
        if (doObj == null) return null;
        TraceRecord entity = new TraceRecord();
        entity.setId(doObj.getId());
        entity.setTraceCode(doObj.getTraceCode());
        entity.setTraceType(doObj.getTraceType() != null ? TraceType.valueOf(doObj.getTraceType()) : null);
        entity.setProductCode(doObj.getProductCode());
        entity.setBatchNo(doObj.getBatchNo());
        entity.setSn(doObj.getSn());
        entity.setSourceTraceCode(doObj.getSourceTraceCode());
        entity.setSource(doObj.getSource() != null ? TraceSource.valueOf(doObj.getSource()) : null);
        entity.setRelations(parseRelations(doObj.getRelations()));
        return entity;
    }

    private TraceRecordDO toDO(TraceRecord entity) {
        if (entity == null) return null;
        TraceRecordDO doObj = new TraceRecordDO();
        doObj.setId(entity.getId());
        doObj.setTraceCode(entity.getTraceCode());
        doObj.setTraceType(entity.getTraceType() != null ? entity.getTraceType().name() : null);
        doObj.setProductCode(entity.getProductCode());
        doObj.setBatchNo(entity.getBatchNo());
        doObj.setSn(entity.getSn());
        doObj.setSourceTraceCode(entity.getSourceTraceCode());
        doObj.setSource(entity.getSource() != null ? entity.getSource().name() : null);
        doObj.setRelations(serializeRelations(entity.getRelations()));
        return doObj;
    }

    private List<TraceRelation> parseRelations(List<Map<String, Object>> raw) {
        if (raw == null) return new ArrayList<>();
        return raw.stream().map(m -> {
            TraceRelation r = new TraceRelation();
            r.setRelatedCode((String) m.get("relatedCode"));
            if (m.get("relatedType") != null) {
                r.setRelatedType(TraceType.valueOf((String) m.get("relatedType")));
            }
            r.setRelationType((String) m.get("relationType"));
            if (m.get("quantity") != null) {
                r.setQuantity((Integer) m.get("quantity"));
            }
            return r;
        }).collect(Collectors.toList());
    }

    private List<Map<String, Object>> serializeRelations(List<TraceRelation> relations) {
        if (relations == null) return Collections.emptyList();
        return relations.stream().map(r -> {
            try {
                return objectMapper.convertValue(r, new TypeReference<Map<String, Object>>() {});
            } catch (Exception e) {
                return Map.<String, Object>of();
            }
        }).collect(Collectors.toList());
    }
}
