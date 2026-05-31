package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryOperationLogDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface InventoryOperationLogMapper {
    void insert(InventoryOperationLogDO log);
    
    List<InventoryOperationLogDO> findByBizId(@Param("bizId") String bizId);
    
    List<InventoryOperationLogDO> findBySkuCodeAndWarehouseId(
        @Param("skuCode") String skuCode, 
        @Param("warehouseId") Long warehouseId);
    
    List<InventoryOperationLogDO> findByOperationType(@Param("operationType") String operationType);
    
    List<InventoryOperationLogDO> findByOperatedAtBetween(
        @Param("start") LocalDateTime start, 
        @Param("end") LocalDateTime end);
    
    List<InventoryOperationLogDO> findByOperatorId(@Param("operatorId") String operatorId);
    
    List<InventoryOperationLogDO> findByResult(@Param("result") String result);
}