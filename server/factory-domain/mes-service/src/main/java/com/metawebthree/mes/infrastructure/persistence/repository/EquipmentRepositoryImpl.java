package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.Equipment;
import com.metawebthree.mes.domain.repository.EquipmentRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.EquipmentMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * 设备仓储实现 - 基于 MyBatis-Plus 持久化
 */
@Repository
public class EquipmentRepositoryImpl implements EquipmentRepository {
    
    @Autowired
    private EquipmentMapper equipmentMapper;
    
    @Override
    public Optional<Equipment> findById(Long id) {
        EquipmentDO equipmentDO = equipmentMapper.selectById(id);
        return Optional.ofNullable(equipmentDO).map(this::toEntity);
    }
    
    @Override
    public Optional<Equipment> findByEquipmentCode(String equipmentCode) {
        LambdaQueryWrapper<EquipmentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentDO::getEquipmentCode, equipmentCode);
        EquipmentDO equipmentDO = equipmentMapper.selectOne(wrapper);
        return Optional.ofNullable(equipmentDO).map(this::toEntity);
    }
    
    @Override
    public List<Equipment> findByWorkshopId(String workshopId) {
        LambdaQueryWrapper<EquipmentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentDO::getWorkshopId, workshopId);
        List<EquipmentDO> doList = equipmentMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<Equipment> findByStatus(Equipment.EquipmentStatus status) {
        LambdaQueryWrapper<EquipmentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentDO::getStatus, status.name());
        List<EquipmentDO> doList = equipmentMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<Equipment> findByWorkstationId(String workstationId) {
        LambdaQueryWrapper<EquipmentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EquipmentDO::getWorkstationId, workstationId);
        List<EquipmentDO> doList = equipmentMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public Equipment save(Equipment equipment) {
        EquipmentDO equipmentDO = toDO(equipment);
        if (equipment.getId() == null) {
            equipmentMapper.insert(equipmentDO);
            equipment.setId(equipmentDO.getId());
        } else {
            equipmentMapper.updateById(equipmentDO);
        }
        return equipment;
    }
    
    @Override
    public void update(Equipment equipment) {
        if (equipment.getId() != null) {
            EquipmentDO equipmentDO = toDO(equipment);
            equipmentMapper.updateById(equipmentDO);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        equipmentMapper.deleteById(id);
    }
    
    private Equipment toEntity(EquipmentDO doObj) {
        if (doObj == null) {
            return null;
        }
        Equipment entity = new Equipment();
        entity.setId(doObj.getId());
        entity.setEquipmentCode(doObj.getEquipmentCode());
        entity.setEquipmentName(doObj.getEquipmentName());
        entity.setEquipmentType(doObj.getEquipmentType());
        entity.setWorkshopId(doObj.getWorkshopId());
        entity.setWorkstationId(doObj.getWorkstationId());
        entity.setStatus(Equipment.EquipmentStatus.valueOf(doObj.getStatus()));
        entity.setUtilizationRate(doObj.getUtilizationRate() != null ? doObj.getUtilizationRate().doubleValue() : null);
        entity.setTodayOutput(doObj.getTodayOutput());
        entity.setCurrentTaskNo(doObj.getCurrentTaskNo());
        entity.setLastMaintenanceTime(doObj.getLastMaintenanceTime());
        entity.setNextMaintenanceTime(doObj.getNextMaintenanceTime());
        return entity;
    }
    
    private EquipmentDO toDO(Equipment entity) {
        if (entity == null) {
            return null;
        }
        EquipmentDO doObj = new EquipmentDO();
        doObj.setId(entity.getId());
        doObj.setEquipmentCode(entity.getEquipmentCode());
        doObj.setEquipmentName(entity.getEquipmentName());
        doObj.setEquipmentType(entity.getEquipmentType());
        doObj.setWorkshopId(entity.getWorkshopId());
        doObj.setWorkstationId(entity.getWorkstationId());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setUtilizationRate(entity.getUtilizationRate() != null ? java.math.BigDecimal.valueOf(entity.getUtilizationRate()) : null);
        doObj.setTodayOutput(entity.getTodayOutput());
        doObj.setCurrentTaskNo(entity.getCurrentTaskNo());
        doObj.setLastMaintenanceTime(entity.getLastMaintenanceTime());
        doObj.setNextMaintenanceTime(entity.getNextMaintenanceTime());
        return doObj;
    }
}