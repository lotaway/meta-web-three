package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.config.WorkOrderType;
import java.util.List;
import java.util.Optional;

public interface WorkOrderTypeRepository {
    Optional<WorkOrderType> findById(Long id);
    Optional<WorkOrderType> findByTypeCode(String typeCode);
    Optional<WorkOrderType> findByIsDefaultTrue();
    List<WorkOrderType> findAll();
    List<WorkOrderType> findByStatus(String status);
    WorkOrderType save(WorkOrderType workOrderType);
    void update(WorkOrderType workOrderType);
    void deleteById(Long id);
}