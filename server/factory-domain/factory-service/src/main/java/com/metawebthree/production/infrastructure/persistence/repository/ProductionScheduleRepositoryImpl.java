package com.metawebthree.production.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.production.domain.entity.ProductionSchedule;
import com.metawebthree.production.domain.repository.ProductionScheduleRepository;
import com.metawebthree.production.infrastructure.persistence.dataobject.ProductionScheduleDO;
import com.metawebthree.production.infrastructure.persistence.mapper.ProductionScheduleMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class ProductionScheduleRepositoryImpl implements ProductionScheduleRepository {
    
    @Autowired
    private ProductionScheduleMapper productionScheduleMapper;
    
    @Override
    public Optional<ProductionSchedule> findById(Long id) {
        ProductionScheduleDO scheduleDO = productionScheduleMapper.selectById(id);
        return Optional.ofNullable(scheduleDO).map(this::toEntity);
    }
    
    @Override
    public Optional<ProductionSchedule> findByScheduleCode(String scheduleCode) {
        LambdaQueryWrapper<ProductionScheduleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionScheduleDO::getScheduleCode, scheduleCode);
        ProductionScheduleDO scheduleDO = productionScheduleMapper.selectOne(wrapper);
        return Optional.ofNullable(scheduleDO).map(this::toEntity);
    }
    
    @Override
    public List<ProductionSchedule> findByOrderCode(String orderCode) {
        LambdaQueryWrapper<ProductionScheduleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionScheduleDO::getOrderCode, orderCode);
        List<ProductionScheduleDO> doList = productionScheduleMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProductionSchedule> findByStationCode(String stationCode) {
        LambdaQueryWrapper<ProductionScheduleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionScheduleDO::getStationCode, stationCode);
        List<ProductionScheduleDO> doList = productionScheduleMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProductionSchedule> findByStatus(ProductionSchedule.ScheduleStatus status) {
        LambdaQueryWrapper<ProductionScheduleDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionScheduleDO::getStatus, status.name());
        List<ProductionScheduleDO> doList = productionScheduleMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProductionSchedule> findAll() {
        List<ProductionScheduleDO> doList = productionScheduleMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public ProductionSchedule save(ProductionSchedule schedule) {
        ProductionScheduleDO scheduleDO = toDO(schedule);
        if (schedule.getId() == null) {
            productionScheduleMapper.insert(scheduleDO);
            schedule.setId(scheduleDO.getId());
        } else {
            productionScheduleMapper.updateById(scheduleDO);
        }
        return schedule;
    }
    
    @Override
    public void delete(ProductionSchedule schedule) {
        if (schedule.getId() != null) {
            productionScheduleMapper.deleteById(schedule.getId());
        }
    }
    
    private ProductionSchedule toEntity(ProductionScheduleDO doObj) {
        if (doObj == null) {
            return null;
        }
        ProductionSchedule entity = new ProductionSchedule();
        entity.setId(doObj.getId());
        entity.setScheduleCode(doObj.getScheduleCode());
        entity.setOrderCode(doObj.getOrderCode());
        entity.setStationCode(doObj.getStationCode());
        entity.setSequence(doObj.getSequence());
        entity.setStatus(doObj.getStatus() != null ? ProductionSchedule.ScheduleStatus.valueOf(doObj.getStatus()) : null);
        entity.setPlannedStartTime(doObj.getPlannedStartTime());
        entity.setPlannedEndTime(doObj.getPlannedEndTime());
        entity.setActualStartTime(doObj.getActualStartTime());
        entity.setActualEndTime(doObj.getActualEndTime());
        entity.setPlannedQuantity(doObj.getPlannedQuantity());
        entity.setCompletedQuantity(doObj.getCompletedQuantity());
        entity.setProgressPercentage(doObj.getProgressPercentage());
        entity.setProcessRouteCode(doObj.getProcessRouteCode());
        entity.setProcessSequence(doObj.getProcessSequence());
        entity.setRequiredSkills(doObj.getRequiredSkills());
        entity.setEstimatedDuration(doObj.getEstimatedDuration());
        entity.setActualDuration(doObj.getActualDuration());
        entity.setNotes(doObj.getNotes());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }
    
    private ProductionScheduleDO toDO(ProductionSchedule entity) {
        if (entity == null) {
            return null;
        }
        ProductionScheduleDO doObj = new ProductionScheduleDO();
        doObj.setId(entity.getId());
        doObj.setScheduleCode(entity.getScheduleCode());
        doObj.setOrderCode(entity.getOrderCode());
        doObj.setStationCode(entity.getStationCode());
        doObj.setSequence(entity.getSequence());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setPlannedStartTime(entity.getPlannedStartTime());
        doObj.setPlannedEndTime(entity.getPlannedEndTime());
        doObj.setActualStartTime(entity.getActualStartTime());
        doObj.setActualEndTime(entity.getActualEndTime());
        doObj.setPlannedQuantity(entity.getPlannedQuantity());
        doObj.setCompletedQuantity(entity.getCompletedQuantity());
        doObj.setProgressPercentage(entity.getProgressPercentage());
        doObj.setProcessRouteCode(entity.getProcessRouteCode());
        doObj.setProcessSequence(entity.getProcessSequence());
        doObj.setRequiredSkills(entity.getRequiredSkills());
        doObj.setEstimatedDuration(entity.getEstimatedDuration());
        doObj.setActualDuration(entity.getActualDuration());
        doObj.setNotes(entity.getNotes());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}