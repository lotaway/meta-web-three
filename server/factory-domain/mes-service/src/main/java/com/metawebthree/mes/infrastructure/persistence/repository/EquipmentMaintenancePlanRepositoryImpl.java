package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.EquipmentMaintenancePlan;
import com.metawebthree.mes.domain.repository.EquipmentMaintenancePlanRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentMaintenancePlanDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentMaintenanceItemDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.EquipmentMaintenancePlanMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.EquipmentMaintenanceItemMapper;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Repository
public class EquipmentMaintenancePlanRepositoryImpl implements EquipmentMaintenancePlanRepository {
    
    private final EquipmentMaintenancePlanMapper planMapper;
    private final EquipmentMaintenanceItemMapper itemMapper;
    
    public EquipmentMaintenancePlanRepositoryImpl(
            EquipmentMaintenancePlanMapper planMapper,
            EquipmentMaintenanceItemMapper itemMapper) {
        this.planMapper = planMapper;
        this.itemMapper = itemMapper;
    }
    
    @Override
    public EquipmentMaintenancePlan save(EquipmentMaintenancePlan plan) {
        EquipmentMaintenancePlanDO doObj = toDO(plan);
        planMapper.insert(doObj);
        plan.setId(doObj.getId());
        
        if (plan.getItems() != null) {
            for (EquipmentMaintenancePlan.MaintenanceItem item : plan.getItems()) {
                EquipmentMaintenanceItemDO itemDO = new EquipmentMaintenanceItemDO();
                itemDO.setPlanId(doObj.getId());
                itemDO.setItemCode(item.getItemCode());
                itemDO.setItemName(item.getItemName());
                itemDO.setDescription(item.getDescription());
                itemDO.setCheckMethod(item.getCheckMethod());
                itemDO.setStandard(item.getStandard());
                itemDO.setIsRequired(item.getIsRequired());
                itemDO.setSortOrder(item.getSortOrder());
                itemMapper.insert(itemDO);
                item.setId(itemDO.getId());
            }
        }
        
        return plan;
    }
    
    @Override
    public Optional<EquipmentMaintenancePlan> findById(Long id) {
        EquipmentMaintenancePlanDO doObj = planMapper.selectById(id);
        if (doObj == null) {
            return Optional.empty();
        }
        return Optional.of(toEntity(doObj));
    }
    
    @Override
    public Optional<EquipmentMaintenancePlan> findByPlanCode(String planCode) {
        EquipmentMaintenancePlanDO doObj = planMapper.selectByPlanCode(planCode);
        if (doObj == null) {
            return Optional.empty();
        }
        return Optional.of(toEntity(doObj));
    }
    
    @Override
    public List<EquipmentMaintenancePlan> findAll() {
        List<EquipmentMaintenancePlanDO> doList = planMapper.selectList(null);
        List<EquipmentMaintenancePlan> result = new ArrayList<>();
        for (EquipmentMaintenancePlanDO doObj : doList) {
            result.add(toEntity(doObj));
        }
        return result;
    }
    
    @Override
    public List<EquipmentMaintenancePlan> findByEquipmentTypeCode(String equipmentTypeCode) {
        List<EquipmentMaintenancePlanDO> doList = planMapper.selectByEquipmentTypeCode(equipmentTypeCode);
        List<EquipmentMaintenancePlan> result = new ArrayList<>();
        for (EquipmentMaintenancePlanDO doObj : doList) {
            result.add(toEntity(doObj));
        }
        return result;
    }
    
    @Override
    public List<EquipmentMaintenancePlan> findByIsActive(Boolean isActive) {
        List<EquipmentMaintenancePlanDO> doList = planMapper.selectByIsActive(isActive);
        List<EquipmentMaintenancePlan> result = new ArrayList<>();
        for (EquipmentMaintenancePlanDO doObj : doList) {
            result.add(toEntity(doObj));
        }
        return result;
    }
    
    @Override
    public void update(EquipmentMaintenancePlan plan) {
        EquipmentMaintenancePlanDO doObj = toDO(plan);
        planMapper.updateById(doObj);
    }
    
    @Override
    public void deleteById(Long id) {
        planMapper.deleteById(id);
    }
    
    private EquipmentMaintenancePlan toEntity(EquipmentMaintenancePlanDO doObj) {
        EquipmentMaintenancePlan plan = new EquipmentMaintenancePlan();
        plan.setId(doObj.getId());
        plan.setPlanCode(doObj.getPlanCode());
        plan.setPlanName(doObj.getPlanName());
        plan.setDescription(doObj.getDescription());
        plan.setEquipmentTypeId(doObj.getEquipmentTypeId());
        plan.setEquipmentTypeCode(doObj.getEquipmentTypeCode());
        plan.setCycleType(EquipmentMaintenancePlan.MaintenanceCycleType.valueOf(doObj.getCycleType()));
        plan.setCycleDays(doObj.getCycleDays());
        plan.setCycleRunningHours(doObj.getCycleRunningHours());
        plan.setAdvanceAlertDays(doObj.getAdvanceAlertDays());
        plan.setIsActive(doObj.getIsActive());
        plan.setCreatedAt(doObj.getCreatedAt());
        plan.setUpdatedAt(doObj.getUpdatedAt());
        
        // 加载保养项目
        List<EquipmentMaintenanceItemDO> itemDOList = itemMapper.selectByPlanId(doObj.getId());
        List<EquipmentMaintenancePlan.MaintenanceItem> items = new ArrayList<>();
        for (EquipmentMaintenanceItemDO itemDO : itemDOList) {
            EquipmentMaintenancePlan.MaintenanceItem item = new EquipmentMaintenancePlan.MaintenanceItem();
            item.setId(itemDO.getId());
            item.setPlanId(itemDO.getPlanId());
            item.setItemCode(itemDO.getItemCode());
            item.setItemName(itemDO.getItemName());
            item.setDescription(itemDO.getDescription());
            item.setCheckMethod(itemDO.getCheckMethod());
            item.setStandard(itemDO.getStandard());
            item.setIsRequired(itemDO.getIsRequired());
            item.setSortOrder(itemDO.getSortOrder());
            items.add(item);
        }
        plan.setItems(items);
        
        return plan;
    }
    
    private EquipmentMaintenancePlanDO toDO(EquipmentMaintenancePlan plan) {
        EquipmentMaintenancePlanDO doObj = new EquipmentMaintenancePlanDO();
        doObj.setId(plan.getId());
        doObj.setPlanCode(plan.getPlanCode());
        doObj.setPlanName(plan.getPlanName());
        doObj.setDescription(plan.getDescription());
        doObj.setEquipmentTypeId(plan.getEquipmentTypeId());
        doObj.setEquipmentTypeCode(plan.getEquipmentTypeCode());
        doObj.setCycleType(plan.getCycleType() != null ? plan.getCycleType().name() : null);
        doObj.setCycleDays(plan.getCycleDays());
        doObj.setCycleRunningHours(plan.getCycleRunningHours());
        doObj.setAdvanceAlertDays(plan.getAdvanceAlertDays());
        doObj.setIsActive(plan.getIsActive());
        doObj.setCreatedAt(plan.getCreatedAt());
        doObj.setUpdatedAt(plan.getUpdatedAt());
        return doObj;
    }
}