package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.repository.ProductionTaskRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProductionTaskDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProductionTaskMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * 生产任务仓储实现 - 基于 MyBatis-Plus 持久化
 */
@Repository
public class ProductionTaskRepositoryImpl implements ProductionTaskRepository {
    
    @Autowired
    private ProductionTaskMapper productionTaskMapper;
    
    @Override
    public Optional<ProductionTask> findById(Long id) {
        ProductionTaskDO taskDO = productionTaskMapper.selectById(id);
        return Optional.ofNullable(taskDO).map(this::toEntity);
    }
    
    @Override
    public Optional<ProductionTask> findByTaskNo(String taskNo) {
        LambdaQueryWrapper<ProductionTaskDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionTaskDO::getTaskNo, taskNo);
        ProductionTaskDO taskDO = productionTaskMapper.selectOne(wrapper);
        return Optional.ofNullable(taskDO).map(this::toEntity);
    }
    
    @Override
    public List<ProductionTask> findByWorkOrderId(Long workOrderId) {
        LambdaQueryWrapper<ProductionTaskDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionTaskDO::getWorkOrderId, workOrderId);
        List<ProductionTaskDO> doList = productionTaskMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProductionTask> findByWorkOrderNo(String workOrderNo) {
        LambdaQueryWrapper<ProductionTaskDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionTaskDO::getWorkOrderNo, workOrderNo);
        List<ProductionTaskDO> doList = productionTaskMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProductionTask> findByStatus(ProductionTask.TaskStatus status) {
        LambdaQueryWrapper<ProductionTaskDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionTaskDO::getStatus, status.name());
        List<ProductionTaskDO> doList = productionTaskMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProductionTask> findByWorkstationId(String workstationId) {
        LambdaQueryWrapper<ProductionTaskDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionTaskDO::getWorkstationId, workstationId);
        List<ProductionTaskDO> doList = productionTaskMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProductionTask> findAll() {
        List<ProductionTaskDO> doList = productionTaskMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public ProductionTask save(ProductionTask task) {
        ProductionTaskDO taskDO = toDO(task);
        if (task.getId() == null) {
            productionTaskMapper.insert(taskDO);
            task.setId(taskDO.getId());
        } else {
            productionTaskMapper.updateById(taskDO);
        }
        return task;
    }
    
    @Override
    public void update(ProductionTask task) {
        if (task.getId() != null) {
            ProductionTaskDO taskDO = toDO(task);
            productionTaskMapper.updateById(taskDO);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        productionTaskMapper.deleteById(id);
    }
    
    private ProductionTask toEntity(ProductionTaskDO doObj) {
        if (doObj == null) {
            return null;
        }
        ProductionTask entity = new ProductionTask();
        entity.setId(doObj.getId());
        entity.setTaskNo(doObj.getTaskNo());
        entity.setWorkOrderId(doObj.getWorkOrderId());
        entity.setWorkstationId(doObj.getWorkstationId());
        entity.setProcessCode(doObj.getStepCode());
        entity.setStatus(ProductionTask.TaskStatus.valueOf(doObj.getStatus()));
        entity.setQuantity(doObj.getPlannedQuantity());
        entity.setCompletedQuantity(doObj.getCompletedQuantity());
        entity.setQualifiedQuantity(doObj.getQualifiedQuantity());
        entity.setDefectiveQuantity(doObj.getRejectedQuantity());
        entity.setOperatorId(doObj.getAssignedTo());
        entity.setStartTime(doObj.getStartTime());
        entity.setEndTime(doObj.getEndTime());
        return entity;
    }
    
    private ProductionTaskDO toDO(ProductionTask entity) {
        if (entity == null) {
            return null;
        }
        ProductionTaskDO doObj = new ProductionTaskDO();
        doObj.setId(entity.getId());
        doObj.setTaskNo(entity.getTaskNo());
        doObj.setWorkOrderId(entity.getWorkOrderId());
        doObj.setWorkstationId(entity.getWorkstationId());
        doObj.setStepCode(entity.getProcessCode());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setPlannedQuantity(entity.getQuantity());
        doObj.setCompletedQuantity(entity.getCompletedQuantity());
        doObj.setQualifiedQuantity(entity.getQualifiedQuantity());
        doObj.setRejectedQuantity(entity.getDefectiveQuantity());
        doObj.setAssignedTo(entity.getOperatorId());
        doObj.setStartTime(entity.getStartTime());
        doObj.setEndTime(entity.getEndTime());
        return doObj;
    }
}