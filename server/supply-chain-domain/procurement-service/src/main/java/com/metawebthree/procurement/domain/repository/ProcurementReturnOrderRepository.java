package com.metawebthree.procurement.domain.repository;

import com.metawebthree.procurement.domain.entity.ProcurementReturnOrder;
import java.util.List;
import java.util.Optional;

public interface ProcurementReturnOrderRepository {
    
    void insert(ProcurementReturnOrder order);
    
    void update(ProcurementReturnOrder order);
    
    Optional<ProcurementReturnOrder> findById(Long id);
    
    Optional<ProcurementReturnOrder> findByReturnNo(String returnNo);
    
    List<ProcurementReturnOrder> findBySupplierCode(String supplierCode);
    
    List<ProcurementReturnOrder> findByWarehouseId(Long warehouseId);
    
    List<ProcurementReturnOrder> findByStatus(String status);
    
    List<ProcurementReturnOrder> findBySourceOrderNo(String sourceOrderNo);
    
    List<ProcurementReturnOrder> findAll();
}