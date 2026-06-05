package com.metawebthree.mes.infrastructure.persistence.repository.labor;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.labor.WorkCenterAssignment;
import com.metawebthree.mes.domain.entity.labor.WorkCenterAssignment.AssignmentStatus;
import com.metawebthree.mes.domain.entity.labor.WorkCenterAssignment.ShiftType;
import com.metawebthree.mes.domain.repository.labor.WorkCenterAssignmentRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.labor.WorkCenterAssignmentDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.labor.WorkCenterAssignmentMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class WorkCenterAssignmentRepositoryImpl implements WorkCenterAssignmentRepository {

    private final WorkCenterAssignmentMapper mapper;

    public WorkCenterAssignmentRepositoryImpl(WorkCenterAssignmentMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<WorkCenterAssignment> findById(Long id) {
        WorkCenterAssignmentDO doObj = mapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }

    @Override
    public List<WorkCenterAssignment> findByOperatorId(Long operatorId) {
        LambdaQueryWrapper<WorkCenterAssignmentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkCenterAssignmentDO::getOperatorId, operatorId);
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<WorkCenterAssignment> findByWorkCenterId(String workCenterId) {
        LambdaQueryWrapper<WorkCenterAssignmentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkCenterAssignmentDO::getWorkCenterId, workCenterId);
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<WorkCenterAssignment> findByStatus(AssignmentStatus status) {
        LambdaQueryWrapper<WorkCenterAssignmentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkCenterAssignmentDO::getStatus, status.name());
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<WorkCenterAssignment> findActiveByOperatorId(Long operatorId) {
        LambdaQueryWrapper<WorkCenterAssignmentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkCenterAssignmentDO::getOperatorId, operatorId);
        wrapper.eq(WorkCenterAssignmentDO::getStatus, AssignmentStatus.ACTIVE.name());
        return mapper.selectList(wrapper).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<WorkCenterAssignment> findAll() {
        return mapper.selectList(null).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public WorkCenterAssignment save(WorkCenterAssignment entity) {
        WorkCenterAssignmentDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            mapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return entity;
    }

    @Override
    public void update(WorkCenterAssignment entity) {
        if (entity.getId() != null) {
            mapper.updateById(toDO(entity));
        }
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private WorkCenterAssignment toEntity(WorkCenterAssignmentDO doObj) {
        if (doObj == null) return null;
        WorkCenterAssignment entity = new WorkCenterAssignment();
        entity.setId(doObj.getId());
        entity.setOperatorId(doObj.getOperatorId());
        entity.setWorkCenterId(doObj.getWorkCenterId());
        entity.setWorkCenterName(doObj.getWorkCenterName());
        entity.setStartDate(doObj.getStartDate());
        entity.setEndDate(doObj.getEndDate());
        entity.setShiftType(doObj.getShiftType() != null ? ShiftType.valueOf(doObj.getShiftType()) : null);
        entity.setStatus(doObj.getStatus() != null ? AssignmentStatus.valueOf(doObj.getStatus()) : AssignmentStatus.ACTIVE);
        entity.setRemark(doObj.getRemark());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private WorkCenterAssignmentDO toDO(WorkCenterAssignment entity) {
        if (entity == null) return null;
        WorkCenterAssignmentDO doObj = new WorkCenterAssignmentDO();
        doObj.setId(entity.getId());
        doObj.setOperatorId(entity.getOperatorId());
        doObj.setWorkCenterId(entity.getWorkCenterId());
        doObj.setWorkCenterName(entity.getWorkCenterName());
        doObj.setStartDate(entity.getStartDate());
        doObj.setEndDate(entity.getEndDate());
        doObj.setShiftType(entity.getShiftType() != null ? entity.getShiftType().name() : null);
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : AssignmentStatus.ACTIVE.name());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
