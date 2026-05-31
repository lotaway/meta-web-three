package com.metawebthree.supplier.domain.repository;

import com.metawebthree.supplier.domain.model.Supplier;
import java.util.List;
import java.util.Optional;

public interface SupplierRepository {

    Supplier save(Supplier supplier);

    Optional<Supplier> findById(Long id);

    Optional<Supplier> findBySupplierCode(String supplierCode);

    List<Supplier> findAll();

    List<Supplier> findByStatus(Supplier.SupplierStatus status);

    List<Supplier> findByVerificationStatus(Supplier.VerificationStatus verificationStatus);

    boolean deleteById(Long id);

    List<Supplier> findBySupplierLevel(Integer level);
}