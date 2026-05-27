package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.QcInspectionType;
import java.util.List;
import java.util.Optional;

public interface QcInspectionTypeRepository {
    
    QcInspectionType save(QcInspectionType entity);
    
    Optional<QcInspectionType> findById(Long id);
    
    Optional<QcInspectionType> findByTypeCode(String typeCode);
    
    List<QcInspectionType> findAll();
    
    List<QcInspectionType> findByCategory(QcInspectionType.InspectionCategory category);
    
    List<QcInspectionType> findByStatus(QcInspectionType.InspectionStatus status);
    
    void deleteById(Long id);
    
    boolean existsByTypeCode(String typeCode);
}