package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.QcPlanItem;
import java.util.List;
import java.util.Optional;

public interface QcPlanItemRepository {
    
    QcPlanItem save(QcPlanItem entity);
    
    Optional<QcPlanItem> findById(Long id);
    
    List<QcPlanItem> findByPlanId(Long planId);
    
    List<QcPlanItem> findByItemId(Long itemId);
    
    void deleteById(Long id);
    
    void deleteByPlanId(Long planId);
}