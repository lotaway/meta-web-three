package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkOrderDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.WorkOrderMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * 工单仓储实现 - 基于 MyBatis-Plus 持久化
 */
@Repository
public class WorkOrderRepositoryImpl implements WorkOrderRepository {
    
    @Autowired
    private WorkOrderMapper workOrderMapper;
    
    @Override
    public Optional<WorkOrder> findById(Long id) {
        WorkOrderDO workOrderDO = workOrderMapper.selectById(id);
        return Optional.ofNullable(workOrderDO).map(this::toEntity);
    }
    
    @Override
    public Optional<WorkOrder> findByWorkOrderNo(String workOrderNo) {
        LambdaQueryWrapper<WorkOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkOrderDO::getWorkOrderNo, workOrderNo);
        WorkOrderDO workOrderDO = workOrderMapper.selectOne(wrapper);
        return Optional.ofNullable(workOrderDO).map(this::toEntity);
    }
    
    @Override
    public List<WorkOrder> findByStatus(WorkOrder.WorkOrderStatus status) {
        LambdaQueryWrapper<WorkOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkOrderDO::getStatus, status.name());
        List<WorkOrderDO> doList = workOrderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkOrder> findByWorkshopId(String workshopId) {
        LambdaQueryWrapper<WorkOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkOrderDO::getWorkshopId, workshopId);
        List<WorkOrderDO> doList = workOrderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkOrder> findByProductCode(String productCode) {
        LambdaQueryWrapper<WorkOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkOrderDO::getProductCode, productCode);
        List<WorkOrderDO> doList = workOrderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkOrder> findByParentWorkOrderId(Long parentWorkOrderId) {
        LambdaQueryWrapper<WorkOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkOrderDO::getParentWorkOrderId, parentWorkOrderId);
        List<WorkOrderDO> doList = workOrderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkOrder> findAll() {
        List<WorkOrderDO> doList = workOrderMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public WorkOrder save(WorkOrder workOrder) {
        WorkOrderDO workOrderDO = toDO(workOrder);
        if (workOrder.getId() == null) {
            workOrderMapper.insert(workOrderDO);
            workOrder.setId(workOrderDO.getId());
        } else {
            workOrderMapper.updateById(workOrderDO);
        }
        return workOrder;
    }
    
    @Override
    public void update(WorkOrder workOrder) {
        if (workOrder.getId() != null) {
            WorkOrderDO workOrderDO = toDO(workOrder);
            workOrderMapper.updateById(workOrderDO);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        workOrderMapper.deleteById(id);
    }
    
    private WorkOrder toEntity(WorkOrderDO doObj) {
        if (doObj == null) {
            return null;
        }
        WorkOrder entity = new WorkOrder();
        entity.setId(doObj.getId());
        entity.setWorkOrderNo(doObj.getWorkOrderNo());
        entity.setProductCode(doObj.getProductCode());
        entity.setProductName(doObj.getProductName());
        entity.setQuantity(doObj.getQuantity());
        entity.setCompletedQuantity(doObj.getCompletedQuantity());
        entity.setStatus(WorkOrder.WorkOrderStatus.valueOf(doObj.getStatus()));
        entity.setStatusCode(doObj.getStatusCode());
        entity.setTypeCode(doObj.getTypeCode());
        entity.setPriority(WorkOrder.Priority.valueOf(doObj.getPriority()));
        entity.setWorkshopId(doObj.getWorkshopId());
        entity.setProcessRouteId(doObj.getProcessRouteId());
        entity.setCodeRuleId(doObj.getCodeRuleId());
        entity.setParentWorkOrderId(doObj.getParentWorkOrderId());
        entity.setSplitRuleId(doObj.getSplitRuleId());
        entity.setSplitSequence(doObj.getSplitSequence());
        entity.setSplitType(doObj.getSplitType());
        entity.setPlannedStartTime(doObj.getPlannedStartTime());
        entity.setPlannedEndTime(doObj.getPlannedEndTime());
        entity.setActualStartTime(doObj.getActualStartTime());
        entity.setActualEndTime(doObj.getActualEndTime());
        return entity;
    }
    
    private WorkOrderDO toDO(WorkOrder entity) {
        if (entity == null) {
            return null;
        }
        WorkOrderDO doObj = new WorkOrderDO();
        doObj.setId(entity.getId());
        doObj.setWorkOrderNo(entity.getWorkOrderNo());
        doObj.setProductCode(entity.getProductCode());
        doObj.setProductName(entity.getProductName());
        doObj.setQuantity(entity.getQuantity());
        doObj.setCompletedQuantity(entity.getCompletedQuantity());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setStatusCode(entity.getStatusCode());
        doObj.setTypeCode(entity.getTypeCode());
        doObj.setPriority(entity.getPriority() != null ? entity.getPriority().name() : null);
        doObj.setWorkshopId(entity.getWorkshopId());
        doObj.setProcessRouteId(entity.getProcessRouteId());
        doObj.setCodeRuleId(entity.getCodeRuleId());
        doObj.setParentWorkOrderId(entity.getParentWorkOrderId());
        doObj.setSplitRuleId(entity.getSplitRuleId());
        doObj.setSplitSequence(entity.getSplitSequence());
        doObj.setSplitType(entity.getSplitType());
        doObj.setPlannedStartTime(entity.getPlannedStartTime());
        doObj.setPlannedEndTime(entity.getPlannedEndTime());
        doObj.setActualStartTime(entity.getActualStartTime());
        doObj.setActualEndTime(entity.getActualEndTime());
        return doObj;
    }
}