package com.metawebthree.supplier.domain.repository;

import com.metawebthree.supplier.domain.entity.Supplier;
import java.util.List;
import java.util.Optional;

public interface SupplierRepository {
    
    Supplier save(Supplier supplier);
    
    Optional<Supplier> findById(Long id);
    
    Optional<Supplier> findByCode(String supplierCode);
    
    List<Supplier> findByStatus(String status);
    
    List<Supplier> findByCategory(String category);
    
    List<Supplier> findByAssessmentLevel(String level);
}