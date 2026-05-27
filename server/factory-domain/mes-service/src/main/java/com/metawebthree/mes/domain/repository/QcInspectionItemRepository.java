package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.QcInspectionItem;
import java.util.List;
import java.util.Optional;

public interface QcInspectionItemRepository {
    
    QcInspectionItem save(QcInspectionItem entity);
    
    Optional<QcInspectionItem> findById(Long id);
    
    Optional<QcInspectionItem> findByItemCode(String itemCode);
    
    List<QcInspectionItem> findAll();
    
    List<QcInspectionItem> findByItemCategory(String itemCategory);
    
    List<QcInspectionItem> findByStatus(QcInspectionItem.ItemStatus status);
    
    void deleteById(Long id);
    
    boolean existsByItemCode(String itemCode);
}