package com.metawebthree.mes.infrastructure.persistence.repository.scada;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.scada.TelemetryMetric;
import com.metawebthree.mes.domain.entity.scada.TelemetryRecord;
import com.metawebthree.mes.domain.repository.scada.TelemetryRecordRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ScadaTelemetryRecordDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ScadaTelemetryRecordMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class TelemetryRecordRepositoryImpl implements TelemetryRecordRepository {

    private final ScadaTelemetryRecordMapper mapper;
    private final ObjectMapper objectMapper;

    public TelemetryRecordRepositoryImpl(ScadaTelemetryRecordMapper mapper, ObjectMapper objectMapper) {
        this.mapper = mapper;
        this.objectMapper = objectMapper;
    }

    @Override
    public Optional<TelemetryRecord> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id)).map(this::toEntity);
    }

    @Override
    public List<TelemetryRecord> findByEquipmentCode(String equipmentCode) {
        LambdaQueryWrapper<ScadaTelemetryRecordDO> w = new LambdaQueryWrapper<>();
        w.eq(ScadaTelemetryRecordDO::getEquipmentCode, equipmentCode)
            .orderByDesc(ScadaTelemetryRecordDO::getCollectTime);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TelemetryRecord> findByEquipmentCodeAndTimeRange(String equipmentCode, LocalDateTime start, LocalDateTime end) {
        LambdaQueryWrapper<ScadaTelemetryRecordDO> w = new LambdaQueryWrapper<>();
        w.eq(ScadaTelemetryRecordDO::getEquipmentCode, equipmentCode)
            .between(ScadaTelemetryRecordDO::getCollectTime, start, end)
            .orderByDesc(ScadaTelemetryRecordDO::getCollectTime);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TelemetryRecord> findByTopic(String topic) {
        LambdaQueryWrapper<ScadaTelemetryRecordDO> w = new LambdaQueryWrapper<>();
        w.eq(ScadaTelemetryRecordDO::getTopic, topic)
            .orderByDesc(ScadaTelemetryRecordDO::getCollectTime);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TelemetryRecord> findAllByTimeRange(LocalDateTime start, LocalDateTime end) {
        LambdaQueryWrapper<ScadaTelemetryRecordDO> w = new LambdaQueryWrapper<>();
        w.between(ScadaTelemetryRecordDO::getCollectTime, start, end)
            .orderByDesc(ScadaTelemetryRecordDO::getCollectTime);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public TelemetryRecord save(TelemetryRecord entity) {
        ScadaTelemetryRecordDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            mapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return entity;
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private TelemetryRecord toEntity(ScadaTelemetryRecordDO doObj) {
        TelemetryRecord entity = new TelemetryRecord();
        entity.setId(doObj.getId());
        entity.setEquipmentCode(doObj.getEquipmentCode());
        entity.setTopic(doObj.getTopic());
        entity.setCollectTime(doObj.getCollectTime());
        entity.setCreatedAt(doObj.getCreatedAt());
        if (doObj.getPayload() != null) {
            try {
                List<TelemetryMetric> metrics = objectMapper.readValue(doObj.getPayload(),
                    objectMapper.getTypeFactory().constructCollectionType(List.class, TelemetryMetric.class));
                entity.setMetrics(metrics);
            } catch (Exception e) {
                entity.setMetrics(Collections.emptyList());
            }
        }
        return entity;
    }

    private ScadaTelemetryRecordDO toDO(TelemetryRecord entity) {
        ScadaTelemetryRecordDO doObj = new ScadaTelemetryRecordDO();
        doObj.setId(entity.getId());
        doObj.setEquipmentCode(entity.getEquipmentCode());
        doObj.setTopic(entity.getTopic());
        doObj.setCollectTime(entity.getCollectTime());
        doObj.setCreatedAt(entity.getCreatedAt());
        if (entity.getMetrics() != null) {
            try {
                doObj.setPayload(objectMapper.writeValueAsString(entity.getMetrics()));
            } catch (JsonProcessingException e) {
                doObj.setPayload("[]");
            }
        }
        return doObj;
    }
}
