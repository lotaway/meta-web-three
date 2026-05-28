package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.NonConformanceDisposition;
import java.util.List;
import java.util.Optional;

public interface NonConformanceDispositionRepository {
    Optional<NonConformanceDisposition> findById(Long id);
    Optional<NonConformanceDisposition> findByDispositionCode(String dispositionCode);
    List<NonConformanceDisposition> findAll();
    List<NonConformanceDisposition> findByType(NonConformanceDisposition.DispositionType type);
    List<NonConformanceDisposition> findByIsEnabled(Boolean isEnabled);
    NonConformanceDisposition save(NonConformanceDisposition disposition);
    void update(NonConformanceDisposition disposition);
    void deleteById(Long id);
    Boolean existsByDispositionCode(String dispositionCode);
}