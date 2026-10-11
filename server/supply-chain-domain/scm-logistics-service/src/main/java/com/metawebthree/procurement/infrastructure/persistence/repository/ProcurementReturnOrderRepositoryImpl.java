package com.metawebthree.procurement.infrastructure.persistence.repository;

import com.metawebthree.procurement.domain.entity.ProcurementReturnOrder;
import com.metawebthree.procurement.domain.entity.ProcurementReturnOrderItem;
import com.metawebthree.procurement.domain.repository.ProcurementReturnOrderRepository;
import com.metawebthree.procurement.infrastructure.persistence.converter.ProcurementReturnOrderConverter;
import com.metawebthree.procurement.infrastructure.persistence.dataobject.ProcurementReturnOrderDO;
import com.metawebthree.procurement.infrastructure.persistence.dataobject.ProcurementReturnOrderItemDO;
import com.metawebthree.procurement.infrastructure.persistence.mapper.ProcurementReturnOrderItemMapper;
import com.metawebthree.procurement.infrastructure.persistence.mapper.ProcurementReturnOrderMapper;
import org.springframework.stereotype.Repository;
import org.springframework.beans.factory.annotation.Autowired;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class ProcurementReturnOrderRepositoryImpl implements ProcurementReturnOrderRepository {
    
    @Autowired
    private ProcurementReturnOrderMapper returnOrderMapper;
    
    @Autowired
    private ProcurementReturnOrderItemMapper returnOrderItemMapper;
    
    @Autowired
    private ProcurementReturnOrderConverter converter;
    
    @Override
    public void insert(ProcurementReturnOrder order) {
        ProcurementReturnOrderDO entity = converter.toDO(order);
        returnOrderMapper.insert(entity);
        order.setId(entity.getId());
        
        if (order.getItems() != null) {
            for (ProcurementReturnOrderItem item : order.getItems()) {
                item.setReturnOrderId(order.getId());
                item.setReturnNo(order.getReturnNo());
                returnOrderItemMapper.insert(toItemDO(item));
            }
        }
    }
    
    @Override
    public void update(ProcurementReturnOrder order) {
        returnOrderMapper.updateById(converter.toDO(order));
        
        if (order.getItems() != null) {
            LambdaQueryWrapper<ProcurementReturnOrderItemDO> wrapper = 
                new LambdaQueryWrapper<>();
            wrapper.eq(ProcurementReturnOrderItemDO::getReturnOrderId, order.getId());
            returnOrderItemMapper.delete(wrapper);
            
            for (ProcurementReturnOrderItem item : order.getItems()) {
                item.setReturnOrderId(order.getId());
                item.setReturnNo(order.getReturnNo());
                returnOrderItemMapper.insert(toItemDO(item));
            }
        }
    }
    
    @Override
    public Optional<ProcurementReturnOrder> findById(Long id) {
        ProcurementReturnOrderDO entity = returnOrderMapper.selectById(id);
        if (entity == null) return Optional.empty();
        
        ProcurementReturnOrder order = converter.toEntity(entity);
        order.setItems(loadItems(order.getId()));
        return Optional.of(order);
    }
    
    @Override
    public Optional<ProcurementReturnOrder> findByReturnNo(String returnNo) {
        LambdaQueryWrapper<ProcurementReturnOrderDO> wrapper = 
            new LambdaQueryWrapper<>();
        wrapper.eq(ProcurementReturnOrderDO::getReturnNo, returnNo);
        ProcurementReturnOrderDO entity = returnOrderMapper.selectOne(wrapper);
        
        if (entity == null) return Optional.empty();
        
        ProcurementReturnOrder order = converter.toEntity(entity);
        order.setItems(loadItems(order.getId()));
        return Optional.of(order);
    }
    
    @Override
    public List<ProcurementReturnOrder> findBySupplierCode(String supplierCode) {
        LambdaQueryWrapper<ProcurementReturnOrderDO> wrapper = 
            new LambdaQueryWrapper<>();
        wrapper.eq(ProcurementReturnOrderDO::getSupplierCode, supplierCode);
        return returnOrderMapper.selectList(wrapper).stream()
            .map(entity -> {
                ProcurementReturnOrder order = converter.toEntity(entity);
                order.setItems(loadItems(order.getId()));
                return order;
            })
            .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcurementReturnOrder> findByWarehouseId(Long warehouseId) {
        LambdaQueryWrapper<ProcurementReturnOrderDO> wrapper = 
            new LambdaQueryWrapper<>();
        wrapper.eq(ProcurementReturnOrderDO::getWarehouseId, warehouseId);
        return returnOrderMapper.selectList(wrapper).stream()
            .map(entity -> {
                ProcurementReturnOrder order = converter.toEntity(entity);
                order.setItems(loadItems(order.getId()));
                return order;
            })
            .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcurementReturnOrder> findByStatus(String status) {
        LambdaQueryWrapper<ProcurementReturnOrderDO> wrapper = 
            new LambdaQueryWrapper<>();
        wrapper.eq(ProcurementReturnOrderDO::getStatus, status);
        return returnOrderMapper.selectList(wrapper).stream()
            .map(entity -> {
                ProcurementReturnOrder order = converter.toEntity(entity);
                order.setItems(loadItems(order.getId()));
                return order;
            })
            .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcurementReturnOrder> findBySourceOrderNo(String sourceOrderNo) {
        LambdaQueryWrapper<ProcurementReturnOrderDO> wrapper = 
            new LambdaQueryWrapper<>();
        wrapper.eq(ProcurementReturnOrderDO::getSourceOrderNo, sourceOrderNo);
        return returnOrderMapper.selectList(wrapper).stream()
            .map(entity -> {
                ProcurementReturnOrder order = converter.toEntity(entity);
                order.setItems(loadItems(order.getId()));
                return order;
            })
            .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcurementReturnOrder> findAll() {
        return returnOrderMapper.selectList(null).stream()
            .map(entity -> {
                ProcurementReturnOrder order = converter.toEntity(entity);
                order.setItems(loadItems(order.getId()));
                return order;
            })
            .collect(Collectors.toList());
    }
    
    private List<ProcurementReturnOrderItem> loadItems(Long returnOrderId) {
        return returnOrderItemMapper.selectByReturnOrderId(returnOrderId).stream()
            .map(this::toItemEntity)
            .collect(Collectors.toList());
    }
    
    private List<ProcurementReturnOrderItem> loadItemsByReturnNo(String returnNo) {
        return returnOrderItemMapper.selectByReturnNo(returnNo).stream()
            .map(this::toItemEntity)
            .collect(Collectors.toList());
    }
    
    private ProcurementReturnOrderItemDO toItemDO(ProcurementReturnOrderItem item) {
        ProcurementReturnOrderItemDO dto = new ProcurementReturnOrderItemDO();
        dto.setId(item.getId());
        dto.setReturnOrderId(item.getReturnOrderId());
        dto.setReturnNo(item.getReturnNo());
        dto.setSourceOrderNo(item.getSourceOrderNo());
        dto.setSourceOrderItemId(item.getSourceOrderItemId());
        dto.setSkuCode(item.getSkuCode());
        dto.setProductName(item.getProductName());
        dto.setReturnQuantity(item.getReturnQuantity());
        dto.setUnitPrice(item.getUnitPrice());
        dto.setTotalAmount(item.getTotalAmount());
        dto.setReason(item.getReason());
        dto.setStatus(item.getStatus());
        return dto;
    }
    
    private ProcurementReturnOrderItem toItemEntity(ProcurementReturnOrderItemDO dto) {
        ProcurementReturnOrderItem item = new ProcurementReturnOrderItem();
        item.setId(dto.getId());
        item.setReturnOrderId(dto.getReturnOrderId());
        item.setReturnNo(dto.getReturnNo());
        item.setSourceOrderNo(dto.getSourceOrderNo());
        item.setSourceOrderItemId(dto.getSourceOrderItemId());
        item.setSkuCode(dto.getSkuCode());
        item.setProductName(dto.getProductName());
        item.setReturnQuantity(dto.getReturnQuantity());
        item.setUnitPrice(dto.getUnitPrice());
        item.setTotalAmount(dto.getTotalAmount());
        item.setReason(dto.getReason());
        item.setStatus(dto.getStatus());
        return item;
    }
}