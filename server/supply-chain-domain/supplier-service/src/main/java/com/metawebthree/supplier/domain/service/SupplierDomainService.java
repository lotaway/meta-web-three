package com.metawebthree.supplier.domain.service;

import com.metawebthree.supplier.domain.entity.Supplier;
import com.metawebthree.supplier.domain.repository.SupplierRepository;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class SupplierDomainService {

    private final SupplierRepository repository;

    public SupplierDomainService(SupplierRepository repository) {
        this.repository = repository;
    }

    public Supplier createSupplier(String code, String name, String category) {
        Supplier supplier = new Supplier();
        supplier.setSupplierCode(code);
        supplier.setName(name);
        supplier.setCategory(category);
        supplier.setStatus("ACTIVE");
        supplier.setAssessmentLevel("B");
        supplier.setCreatedAt(java.time.LocalDateTime.now());
        return repository.save(supplier);
    }

    public Optional<Supplier> findById(Long id) {
        return repository.findById(id);
    }

    public Optional<Supplier> findByCode(String supplierCode) {
        return repository.findByCode(supplierCode);
    }

    public Supplier updateAssessment(Long id, String level) {
        return repository.findById(id)
            .map(s -> {
                s.setAssessmentLevel(level);
                s.setUpdatedAt(java.time.LocalDateTime.now());
                return repository.save(s);
            })
            .orElseThrow(() -> new IllegalArgumentException("Supplier not found"));
    }

    public Supplier updateStatus(Long id, String status) {
        return repository.findById(id)
            .map(s -> {
                s.setStatus(status);
                s.setUpdatedAt(java.time.LocalDateTime.now());
                return repository.save(s);
            })
            .orElseThrow(() -> new IllegalArgumentException("Supplier not found"));
    }
}