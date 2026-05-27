package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.EquipmentChecklistRecord;
import com.metawebthree.mes.domain.repository.EquipmentChecklistRecordRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentChecklistRecordDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.EquipmentChecklistRecordMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Repository
public class EquipmentChecklistRecordRepositoryImpl implements EquipmentChecklistRecordRepository {
    
    @Autowired
    private EquipmentChecklistRecordMapper recordMapper;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Override
    public Optional<EquipmentChecklistRecord> findById(Long id) {
        EquipmentChecklistRecordDO recordDO = recordMapper.selectById(id);
        return Optional.ofNullable(recordDO).map(this::toEntity);
    }
    
    @Override
    public Optional<EquipmentChecklistRecord> findByRecordCode(String recordCode) {
        LambdaQueryWrapper<EquipmentChecklistRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistRecordDO::getRecordCode, recordCode);
        EquipmentChecklistRecordDO recordDO = recordMapper.selectOne(wrapper);
        return Optional.ofNullable(recordDO).map(this::toEntity);
    }
    
    @Override
    public List<EquipmentChecklistRecord> findByEquipmentId(Long equipmentId) {
        LambdaQueryWrapper<EquipmentChecklistRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistRecordDO::getEquipmentId, equipmentId);
        wrapper.orderByDesc(EquipmentChecklistRecordDO::getCheckPlanTime);
        List<EquipmentChecklistRecordDO> doList = recordMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<EquipmentChecklistRecord> findByEquipmentCode(String equipmentCode) {
        LambdaQueryWrapper<EquipmentChecklistRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistRecordDO::getEquipmentCode, equipmentCode);
        wrapper.orderByDesc(EquipmentChecklistRecordDO::getCheckPlanTime);
        List<EquipmentChecklistRecordDO> doList = recordMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<EquipmentChecklistRecord> findByTemplateId(Long templateId) {
        LambdaQueryWrapper<EquipmentChecklistRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistRecordDO::getTemplateId, templateId);
        wrapper.orderByDesc(EquipmentChecklistRecordDO::getCheckPlanTime);
        List<EquipmentChecklistRecordDO> doList = recordMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<EquipmentChecklistRecord> findByStatus(EquipmentChecklistRecord.RecordStatus status) {
        LambdaQueryWrapper<EquipmentChecklistRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistRecordDO::getStatus, status.name());
        List<EquipmentChecklistRecordDO> doList = recordMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<EquipmentChecklistRecord> findByCheckerId(String checkerId) {
        LambdaQueryWrapper<EquipmentChecklistRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistRecordDO::getCheckerId, checkerId);
        wrapper.orderByDesc(EquipmentChecklistRecordDO::getCheckPlanTime);
        List<EquipmentChecklistRecordDO> doList = recordMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<EquipmentChecklistRecord> findOverdueRecords() {
        LocalDateTime yesterday = LocalDateTime.now().minusHours(24);
        LambdaQueryWrapper<EquipmentChecklistRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistRecordDO::getStatus, EquipmentChecklistRecord.RecordStatus.PENDING.name());
        wrapper.lt(EquipmentChecklistRecordDO::getCheckPlanTime, yesterday);
        List<EquipmentChecklistRecordDO> doList = recordMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<EquipmentChecklistRecord> findByEquipmentIdAndCheckPlanTimeBetween(
            Long equipmentId, LocalDateTime startTime, LocalDateTime endTime) {
        LambdaQueryWrapper<EquipmentChecklistRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentChecklistRecordDO::getEquipmentId, equipmentId);
        wrapper.between(EquipmentChecklistRecordDO::getCheckPlanTime, startTime, endTime);
        wrapper.orderByDesc(EquipmentChecklistRecordDO::getCheckPlanTime);
        List<EquipmentChecklistRecordDO> doList = recordMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public EquipmentChecklistRecord save(EquipmentChecklistRecord record) {
        EquipmentChecklistRecordDO recordDO = toDO(record);
        if (record.getId() == null) {
            recordMapper.insert(recordDO);
            record.setId(recordDO.getId());
        } else {
            recordMapper.updateById(recordDO);
        }
        return record;
    }
    
    @Override
    public void update(EquipmentChecklistRecord record) {
        recordMapper.updateById(toDO(record));
    }
    
    @Override
    public void deleteById(Long id) {
        recordMapper.deleteById(id);
    }
    
    private EquipmentChecklistRecord toEntity(EquipmentChecklistRecordDO doObj) {
        if (doObj == null) {
            return null;
        }
        EquipmentChecklistRecord record = new EquipmentChecklistRecord();
        record.setId(doObj.getId());
        record.setRecordCode(doObj.getRecordCode());
        record.setEquipmentId(doObj.getEquipmentId());
        record.setEquipmentCode(doObj.getEquipmentCode());
        record.setTemplateId(doObj.getTemplateId());
        record.setTemplateCode(doObj.getTemplateCode());
        record.setCheckPlanTime(doObj.getCheckPlanTime());
        record.setCheckActualTime(doObj.getCheckActualTime());
        record.setCheckerId(doObj.getCheckerId());
        record.setCheckerName(doObj.getCheckerName());
        record.setStatus(doObj.getStatus() != null ? EquipmentChecklistRecord.RecordStatus.valueOf(doObj.getStatus()) : null);
        record.setTotalItems(doObj.getTotalItems());
        record.setCheckedItems(doObj.getCheckedItems());
        record.setAbnormalItems(doObj.getAbnormalItems());
        record.setCheckResult(doObj.getCheckResult());
        record.setRemark(doObj.getRemark());
        record.setCreatedAt(doObj.getCreatedAt());
        record.setUpdatedAt(doObj.getUpdatedAt());
        
        // 反序列化 itemResults
        if (doObj.getItemResults() != null && !doObj.getItemResults().isEmpty()) {
            try {
                Map<String, Object> itemResults = objectMapper.readValue(
                    doObj.getItemResults(), 
                    new TypeReference<Map<String, Object>>() {}
                );
                record.setItemResults(itemResults);
            } catch (JsonProcessingException e) {
                // 忽略反序列化错误
            }
        }
        
        return record;
    }
    
    private EquipmentChecklistRecordDO toDO(EquipmentChecklistRecord record) {
        EquipmentChecklistRecordDO doObj = new EquipmentChecklistRecordDO();
        doObj.setId(record.getId());
        doObj.setRecordCode(record.getRecordCode());
        doObj.setEquipmentId(record.getEquipmentId());
        doObj.setEquipmentCode(record.getEquipmentCode());
        doObj.setTemplateId(record.getTemplateId());
        doObj.setTemplateCode(record.getTemplateCode());
        doObj.setCheckPlanTime(record.getCheckPlanTime());
        doObj.setCheckActualTime(record.getCheckActualTime());
        doObj.setCheckerId(record.getCheckerId());
        doObj.setCheckerName(record.getCheckerName());
        doObj.setStatus(record.getStatus() != null ? record.getStatus().name() : null);
        doObj.setTotalItems(record.getTotalItems());
        doObj.setCheckedItems(record.getCheckedItems());
        doObj.setAbnormalItems(record.getAbnormalItems());
        doObj.setCheckResult(record.getCheckResult());
        doObj.setRemark(record.getRemark());
        doObj.setCreatedAt(record.getCreatedAt());
        doObj.setUpdatedAt(record.getUpdatedAt());
        
        // 序列化 itemResults
        if (record.getItemResults() != null && !record.getItemResults().isEmpty()) {
            try {
                doObj.setItemResults(objectMapper.writeValueAsString(record.getItemResults()));
            } catch (JsonProcessingException e) {
                doObj.setItemResults("{}");
            }
        }
        
        return doObj;
    }
}