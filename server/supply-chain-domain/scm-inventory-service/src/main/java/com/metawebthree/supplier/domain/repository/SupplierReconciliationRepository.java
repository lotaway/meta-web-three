package com.metawebthree.supplier.domain.repository;

import com.metawebthree.supplier.domain.entity.SupplierReconciliation;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface SupplierReconciliationRepository {
    
    SupplierReconciliation save(SupplierReconciliation reconciliation);
    
    Optional<SupplierReconciliation> findById(Long id);
    
    Optional<SupplierReconciliation> findByReconciliationNo(String reconciliationNo);
    
    List<SupplierReconciliation> findBySupplierCode(String supplierCode);
    
    List<SupplierReconciliation> findBySupplierCodeAndStatus(String supplierCode, String status);
    
    List<SupplierReconciliation> findByPeriod(String supplierCode, LocalDate periodStart, LocalDate periodEnd);
    
    List<SupplierReconciliation> findByStatus(String status);
    
    void deleteById(Long id);
}