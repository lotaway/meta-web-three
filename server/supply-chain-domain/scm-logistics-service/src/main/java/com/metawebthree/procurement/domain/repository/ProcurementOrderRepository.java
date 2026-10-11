package com.metawebthree.procurement.domain.repository;

import com.metawebthree.procurement.domain.entity.ProcurementOrder;
import java.util.Optional;
import java.util.List;

public interface ProcurementOrderRepository {
    
    ProcurementOrder save(ProcurementOrder order);
    
    Optional<ProcurementOrder> findById(Long id);
    
    Optional<ProcurementOrder> findByOrderNo(String orderNo);
    
    List<ProcurementOrder> findByStatus(String status);
    
    List<ProcurementOrder> findBySupplierCode(String supplierCode);
    
    List<ProcurementOrder> findByWarehouseId(Long warehouseId);
}