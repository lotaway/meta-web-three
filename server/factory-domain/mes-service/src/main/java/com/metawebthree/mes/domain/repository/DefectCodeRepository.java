package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.DefectCode;
import java.util.List;
import java.util.Optional;

public interface DefectCodeRepository {
    Optional<DefectCode> findById(Long id);
    Optional<DefectCode> findByDefectCode(String defectCode);
    List<DefectCode> findAll();
    List<DefectCode> findByCategory(DefectCode.DefectCategory category);
    List<DefectCode> findBySeverity(DefectCode.DefectSeverity severity);
    List<DefectCode> findByIsEnabled(Boolean isEnabled);
    DefectCode save(DefectCode defectCode);
    void update(DefectCode defectCode);
    void deleteById(Long id);
    Boolean existsByDefectCode(String defectCode);
}