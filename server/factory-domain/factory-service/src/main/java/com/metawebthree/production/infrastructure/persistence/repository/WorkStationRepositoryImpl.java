package com.metawebthree.production.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.production.domain.entity.WorkStation;
import com.metawebthree.production.domain.repository.WorkStationRepository;
import com.metawebthree.production.infrastructure.persistence.dataobject.WorkStationDO;
import com.metawebthree.production.infrastructure.persistence.mapper.WorkStationMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class WorkStationRepositoryImpl implements WorkStationRepository {
    
    @Autowired
    private WorkStationMapper workStationMapper;
    
    @Override
    public Optional<WorkStation> findById(Long id) {
        WorkStationDO stationDO = workStationMapper.selectById(id);
        return Optional.ofNullable(stationDO).map(this::toEntity);
    }
    
    @Override
    public Optional<WorkStation> findByStationCode(String stationCode) {
        LambdaQueryWrapper<WorkStationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkStationDO::getStationCode, stationCode);
        WorkStationDO stationDO = workStationMapper.selectOne(wrapper);
        return Optional.ofNullable(stationDO).map(this::toEntity);
    }
    
    @Override
    public List<WorkStation> findByStatus(WorkStation.StationStatus status) {
        LambdaQueryWrapper<WorkStationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkStationDO::getStatus, status.name());
        List<WorkStationDO> doList = workStationMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkStation> findByWorkshopCode(String workshopCode) {
        LambdaQueryWrapper<WorkStationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkStationDO::getWorkshopCode, workshopCode);
        List<WorkStationDO> doList = workStationMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkStation> findAll() {
        List<WorkStationDO> doList = workStationMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public WorkStation save(WorkStation station) {
        WorkStationDO stationDO = toDO(station);
        if (station.getId() == null) {
            workStationMapper.insert(stationDO);
            station.setId(stationDO.getId());
        } else {
            workStationMapper.updateById(stationDO);
        }
        return station;
    }
    
    @Override
    public void delete(WorkStation station) {
        if (station.getId() != null) {
            workStationMapper.deleteById(station.getId());
        }
    }
    
    @Override
    public List<WorkStation> findAvailableStations() {
        LambdaQueryWrapper<WorkStationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkStationDO::getStatus, WorkStation.StationStatus.IDLE.name());
        List<WorkStationDO> doList = workStationMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    private WorkStation toEntity(WorkStationDO doObj) {
        if (doObj == null) {
            return null;
        }
        WorkStation entity = new WorkStation();
        entity.setId(doObj.getId());
        entity.setStationCode(doObj.getStationCode());
        entity.setStationName(doObj.getStationName());
        entity.setStationType(doObj.getStationType());
        entity.setWorkshopCode(doObj.getWorkshopCode());
        entity.setProductionLineCode(doObj.getProductionLineCode());
        entity.setStatus(doObj.getStatus() != null ? WorkStation.StationStatus.valueOf(doObj.getStatus()) : null);
        entity.setCapacity(doObj.getCapacity());
        entity.setCurrentLoad(doObj.getCurrentLoad());
        entity.setEfficiency(doObj.getEfficiency());
        entity.setCurrentOperator(doObj.getCurrentOperator());
        entity.setCurrentOrderCode(doObj.getCurrentOrderCode());
        entity.setPositionX(doObj.getPositionX());
        entity.setPositionY(doObj.getPositionY());
        entity.setIpAddress(doObj.getIpAddress());
        entity.setLastMaintenanceTime(doObj.getLastMaintenanceTime());
        entity.setNextMaintenanceTime(doObj.getNextMaintenanceTime());
        entity.setEquipmentList(doObj.getEquipmentList());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }
    
    private WorkStationDO toDO(WorkStation entity) {
        if (entity == null) {
            return null;
        }
        WorkStationDO doObj = new WorkStationDO();
        doObj.setId(entity.getId());
        doObj.setStationCode(entity.getStationCode());
        doObj.setStationName(entity.getStationName());
        doObj.setStationType(entity.getStationType());
        doObj.setWorkshopCode(entity.getWorkshopCode());
        doObj.setProductionLineCode(entity.getProductionLineCode());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setCapacity(entity.getCapacity());
        doObj.setCurrentLoad(entity.getCurrentLoad());
        doObj.setEfficiency(entity.getEfficiency());
        doObj.setCurrentOperator(entity.getCurrentOperator());
        doObj.setCurrentOrderCode(entity.getCurrentOrderCode());
        doObj.setPositionX(entity.getPositionX());
        doObj.setPositionY(entity.getPositionY());
        doObj.setIpAddress(entity.getIpAddress());
        doObj.setLastMaintenanceTime(entity.getLastMaintenanceTime());
        doObj.setNextMaintenanceTime(entity.getNextMaintenanceTime());
        doObj.setEquipmentList(entity.getEquipmentList());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}