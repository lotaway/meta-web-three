package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.InventoryOperationLog;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryOperationLogDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.InventoryOperationLogMapper;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Repository
public class InventoryOperationLogRepositoryImpl implements InventoryOperationLogRepository {
    
    private final InventoryOperationLogMapper mapper;
    
    public InventoryOperationLogRepositoryImpl(InventoryOperationLogMapper mapper) {
        this.mapper = mapper;
    }
    
    @Override
    public InventoryOperationLog save(InventoryOperationLog log) {
        if (log.getOperatedAt() == null) {
            log.setOperatedAt(LocalDateTime.now());
        }
        mapper.insert(toDO(log));
        return log;
    }
    
    @Override
    public List<InventoryOperationLog> findByBizId(String bizId) {
        return mapper.findByBizId(bizId).stream()
            .map(this::toEntity)
            .collect(Collectors.toList());
    }
    
    @Override
    public List<InventoryOperationLog> findBySkuCodeAndWarehouseId(String skuCode, Long warehouseId) {
        return mapper.findBySkuCodeAndWarehouseId(skuCode, warehouseId).stream()
            .map(this::toEntity)
            .collect(Collectors.toList());
    }
    
    @Override
    public List<InventoryOperationLog> findByOperationType(String operationType) {
        return mapper.findByOperationType(operationType).stream()
            .map(this::toEntity)
            .collect(Collectors.toList());
    }
    
    @Override
    public List<InventoryOperationLog> findByOperatedAtBetween(LocalDateTime start, LocalDateTime end) {
        return mapper.findByOperatedAtBetween(start, end).stream()
            .map(this::toEntity)
            .collect(Collectors.toList());
    }
    
    @Override
    public List<InventoryOperationLog> findByOperatorId(String operatorId) {
        return mapper.findByOperatorId(operatorId).stream()
            .map(this::toEntity)
            .collect(Collectors.toList());
    }
    
    @Override
    public List<InventoryOperationLog> findByResult(String result) {
        return mapper.findByResult(result).stream()
            .map(this::toEntity)
            .collect(Collectors.toList());
    }
    
    private InventoryOperationLogDO toDO(InventoryOperationLog entity) {
        InventoryOperationLogDO doObj = new InventoryOperationLogDO();
        doObj.setOperationType(entity.getOperationType());
        doObj.setSkuCode(entity.getSkuCode());
        doObj.setWarehouseId(entity.getWarehouseId());
        doObj.setQuantity(entity.getQuantity());
        doObj.setBizId(entity.getBizId());
        doObj.setRemark(entity.getRemark());
        doObj.setOperatorId(entity.getOperatorId());
        doObj.setOperatorName(entity.getOperatorName());
        doObj.setQuantityBefore(entity.getQuantityBefore());
        doObj.setQuantityAfter(entity.getQuantityAfter());
        doObj.setOperatedAt(entity.getOperatedAt());
        doObj.setResult(entity.getResult());
        doObj.setErrorMessage(entity.getErrorMessage());
        doObj.setRequestId(entity.getRequestId());
        doObj.setClientIp(entity.getClientIp());
        return doObj;
    }
    
    private InventoryOperationLog toEntity(InventoryOperationLogDO doObj) {
        InventoryOperationLog entity = new InventoryOperationLog();
        entity.setId(doObj.getId());
        entity.setOperationType(doObj.getOperationType());
        entity.setSkuCode(doObj.getSkuCode());
        entity.setWarehouseId(doObj.getWarehouseId());
        entity.setQuantity(doObj.getQuantity());
        entity.setBizId(doObj.getBizId());
        entity.setRemark(doObj.getRemark());
        entity.setOperatorId(doObj.getOperatorId());
        entity.setOperatorName(doObj.getOperatorName());
        entity.setQuantityBefore(doObj.getQuantityBefore());
        entity.setQuantityAfter(doObj.getQuantityAfter());
        entity.setOperatedAt(doObj.getOperatedAt());
        entity.setResult(doObj.getResult());
        entity.setErrorMessage(doObj.getErrorMessage());
        entity.setRequestId(doObj.getRequestId());
        entity.setClientIp(doObj.getClientIp());
        return entity;
    }
}