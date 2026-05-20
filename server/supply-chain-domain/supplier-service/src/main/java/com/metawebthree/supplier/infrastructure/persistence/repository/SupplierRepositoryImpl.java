package com.metawebthree.supplier.infrastructure.persistence.repository;

import com.metawebthree.supplier.domain.entity.Supplier;
import com.metawebthree.supplier.domain.repository.SupplierRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Repository
public class SupplierRepositoryImpl implements SupplierRepository {

    private final Map<Long, Supplier> store = new ConcurrentHashMap<>();
    private final Map<String, Supplier> codeIndex = new ConcurrentHashMap<>();
    private Long idSequence = 1L;

    @Override
    public Supplier save(Supplier supplier) {
        if (supplier.getId() == null) {
            supplier.setId(idSequence++);
            supplier.setCreatedAt(java.time.LocalDateTime.now());
        }
        supplier.setUpdatedAt(java.time.LocalDateTime.now());
        store.put(supplier.getId(), supplier);
        codeIndex.put(supplier.getSupplierCode(), supplier);
        return supplier;
    }

    @Override
    public Optional<Supplier> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public Optional<Supplier> findByCode(String supplierCode) {
        return Optional.ofNullable(codeIndex.get(supplierCode));
    }

    @Override
    public List<Supplier> findByStatus(String status) {
        return store.values().stream()
            .filter(s -> status.equals(s.getStatus()))
            .toList();
    }

    @Override
    public List<Supplier> findByCategory(String category) {
        return store.values().stream()
            .filter(s -> category.equals(s.getCategory()))
            .toList();
    }

    @Override
    public List<Supplier> findByAssessmentLevel(String level) {
        return store.values().stream()
            .filter(s -> level.equals(s.getAssessmentLevel()))
            .toList();
    }
}