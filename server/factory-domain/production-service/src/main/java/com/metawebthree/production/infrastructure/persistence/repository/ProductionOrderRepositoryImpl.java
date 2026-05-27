package com.metawebthree.production.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.production.domain.entity.ProductionOrder;
import com.metawebthree.production.domain.repository.ProductionOrderRepository;
import com.metawebthree.production.infrastructure.persistence.dataobject.ProductionOrderDO;
import com.metawebthree.production.infrastructure.persistence.mapper.ProductionOrderMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class ProductionOrderRepositoryImpl implements ProductionOrderRepository {
    
    @Autowired
    private ProductionOrderMapper productionOrderMapper;
    
    @Override
    public Optional<ProductionOrder> findById(Long id) {
        ProductionOrderDO orderDO = productionOrderMapper.selectById(id);
        return Optional.ofNullable(orderDO).map(this::toEntity);
    }
    
    @Override
    public Optional<ProductionOrder> findByOrderCode(String orderCode) {
        LambdaQueryWrapper<ProductionOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionOrderDO::getOrderCode, orderCode);
        ProductionOrderDO orderDO = productionOrderMapper.selectOne(wrapper);
        return Optional.ofNullable(orderDO).map(this::toEntity);
    }
    
    @Override
    public List<ProductionOrder> findByStatus(ProductionOrder.OrderStatus status) {
        LambdaQueryWrapper<ProductionOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionOrderDO::getStatus, status.name());
        List<ProductionOrderDO> doList = productionOrderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProductionOrder> findByWorkshopCode(String workshopCode) {
        LambdaQueryWrapper<ProductionOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionOrderDO::getWorkshopCode, workshopCode);
        List<ProductionOrderDO> doList = productionOrderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProductionOrder> findAll() {
        List<ProductionOrderDO> doList = productionOrderMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public ProductionOrder save(ProductionOrder order) {
        ProductionOrderDO orderDO = toDO(order);
        if (order.getId() == null) {
            productionOrderMapper.insert(orderDO);
            order.setId(orderDO.getId());
        } else {
            productionOrderMapper.updateById(orderDO);
        }
        return order;
    }
    
    @Override
    public void delete(ProductionOrder order) {
        if (order.getId() != null) {
            productionOrderMapper.deleteById(order.getId());
        }
    }
    
    @Override
    public List<ProductionOrder> findByPriority(ProductionOrder.Priority priority) {
        LambdaQueryWrapper<ProductionOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProductionOrderDO::getPriority, priority.name());
        List<ProductionOrderDO> doList = productionOrderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    private ProductionOrder toEntity(ProductionOrderDO doObj) {
        if (doObj == null) {
            return null;
        }
        ProductionOrder entity = new ProductionOrder();
        entity.setId(doObj.getId());
        entity.setOrderCode(doObj.getOrderCode());
        entity.setProductCode(doObj.getProductCode());
        entity.setProductName(doObj.getProductName());
        entity.setQuantityPlanned(doObj.getQuantityPlanned());
        entity.setQuantityCompleted(doObj.getQuantityCompleted());
        entity.setStatus(doObj.getStatus() != null ? ProductionOrder.OrderStatus.valueOf(doObj.getStatus()) : null);
        entity.setPriority(doObj.getPriority() != null ? ProductionOrder.Priority.valueOf(doObj.getPriority()) : null);
        entity.setWorkshopCode(doObj.getWorkshopCode());
        entity.setProductionLineCode(doObj.getProductionLineCode());
        entity.setPlannedStartTime(doObj.getPlannedStartTime());
        entity.setPlannedEndTime(doObj.getPlannedEndTime());
        entity.setActualStartTime(doObj.getActualStartTime());
        entity.setActualEndTime(doObj.getActualEndTime());
        entity.setProgressPercentage(doObj.getProgressPercentage());
        entity.setOrderType(doObj.getOrderType());
        entity.setCustomerName(doObj.getCustomerName());
        entity.setNotes(doObj.getNotes());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }
    
    private ProductionOrderDO toDO(ProductionOrder entity) {
        if (entity == null) {
            return null;
        }
        ProductionOrderDO doObj = new ProductionOrderDO();
        doObj.setId(entity.getId());
        doObj.setOrderCode(entity.getOrderCode());
        doObj.setProductCode(entity.getProductCode());
        doObj.setProductName(entity.getProductName());
        doObj.setQuantityPlanned(entity.getQuantityPlanned());
        doObj.setQuantityCompleted(entity.getQuantityCompleted());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setPriority(entity.getPriority() != null ? entity.getPriority().name() : null);
        doObj.setWorkshopCode(entity.getWorkshopCode());
        doObj.setProductionLineCode(entity.getProductionLineCode());
        doObj.setPlannedStartTime(entity.getPlannedStartTime());
        doObj.setPlannedEndTime(entity.getPlannedEndTime());
        doObj.setActualStartTime(entity.getActualStartTime());
        doObj.setActualEndTime(entity.getActualEndTime());
        doObj.setProgressPercentage(entity.getProgressPercentage());
        doObj.setOrderType(entity.getOrderType());
        doObj.setCustomerName(entity.getCustomerName());
        doObj.setNotes(entity.getNotes());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}