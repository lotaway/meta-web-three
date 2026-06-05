package com.metawebthree.mes.infrastructure.persistence.repository.labor;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.labor.TimeRecord;
import com.metawebthree.mes.domain.entity.labor.TimeRecord.RecordStatus;
import com.metawebthree.mes.domain.entity.labor.TimeRecord.RecordType;
import com.metawebthree.mes.domain.repository.labor.TimeRecordRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.labor.TimeRecordDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.labor.TimeRecordMapper;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class TimeRecordRepositoryImpl implements TimeRecordRepository {

    private final TimeRecordMapper mapper;

    public TimeRecordRepositoryImpl(TimeRecordMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<TimeRecord> findById(Long id) {
        TimeRecordDO doObj = mapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }

    @Override
    public List<TimeRecord> findByOperatorId(Long operatorId) {
        LambdaQueryWrapper<TimeRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(TimeRecordDO::getOperatorId, operatorId);
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TimeRecord> findByOperatorIdAndDate(Long operatorId, LocalDate date) {
        LambdaQueryWrapper<TimeRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(TimeRecordDO::getOperatorId, operatorId);
        wrapper.eq(TimeRecordDO::getRecordDate, date);
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TimeRecord> findByDateRange(LocalDate start, LocalDate end) {
        LambdaQueryWrapper<TimeRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(TimeRecordDO::getRecordDate, start, end);
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TimeRecord> findByWorkOrderNo(String workOrderNo) {
        LambdaQueryWrapper<TimeRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(TimeRecordDO::getWorkOrderNo, workOrderNo);
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TimeRecord> findByStatus(RecordStatus status) {
        LambdaQueryWrapper<TimeRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(TimeRecordDO::getStatus, status.name());
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<TimeRecord> findAll() {
        return mapper.selectList(null).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public TimeRecord save(TimeRecord entity) {
        TimeRecordDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            mapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return entity;
    }

    @Override
    public void update(TimeRecord entity) {
        if (entity.getId() != null) {
            mapper.updateById(toDO(entity));
        }
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private TimeRecord toEntity(TimeRecordDO doObj) {
        if (doObj == null) return null;
        TimeRecord entity = new TimeRecord();
        entity.setId(doObj.getId());
        entity.setOperatorId(doObj.getOperatorId());
        entity.setOperatorCode(doObj.getOperatorCode());
        entity.setOperatorName(doObj.getOperatorName());
        entity.setWorkOrderNo(doObj.getWorkOrderNo());
        entity.setTaskNo(doObj.getTaskNo());
        entity.setOperationCode(doObj.getOperationCode());
        entity.setWorkCenterId(doObj.getWorkCenterId());
        entity.setRecordDate(doObj.getRecordDate());
        entity.setStartTime(doObj.getStartTime());
        entity.setEndTime(doObj.getEndTime());
        entity.setTotalHours(doObj.getTotalHours() != null ? doObj.getTotalHours() : BigDecimal.ZERO);
        entity.setRecordType(doObj.getRecordType() != null ? RecordType.valueOf(doObj.getRecordType()) : RecordType.REGULAR);
        entity.setStatus(doObj.getStatus() != null ? RecordStatus.valueOf(doObj.getStatus()) : RecordStatus.DRAFT);
        entity.setApprovedBy(doObj.getApprovedBy());
        entity.setApprovedAt(doObj.getApprovedAt());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private TimeRecordDO toDO(TimeRecord entity) {
        if (entity == null) return null;
        TimeRecordDO doObj = new TimeRecordDO();
        doObj.setId(entity.getId());
        doObj.setOperatorId(entity.getOperatorId());
        doObj.setOperatorCode(entity.getOperatorCode());
        doObj.setOperatorName(entity.getOperatorName());
        doObj.setWorkOrderNo(entity.getWorkOrderNo());
        doObj.setTaskNo(entity.getTaskNo());
        doObj.setOperationCode(entity.getOperationCode());
        doObj.setWorkCenterId(entity.getWorkCenterId());
        doObj.setRecordDate(entity.getRecordDate());
        doObj.setStartTime(entity.getStartTime());
        doObj.setEndTime(entity.getEndTime());
        doObj.setTotalHours(entity.getTotalHours());
        doObj.setRecordType(entity.getRecordType() != null ? entity.getRecordType().name() : null);
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : RecordStatus.DRAFT.name());
        doObj.setApprovedBy(entity.getApprovedBy());
        doObj.setApprovedAt(entity.getApprovedAt());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
