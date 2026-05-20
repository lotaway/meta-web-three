package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.ProductionTask;
import java.util.List;
import java.util.Optional;

public interface ProductionTaskRepository {
    Optional<ProductionTask> findById(Long id);
    Optional<ProductionTask> findByTaskNo(String taskNo);
    List<ProductionTask> findByWorkOrderId(Long workOrderId);
    List<ProductionTask> findByStatus(ProductionTask.TaskStatus status);
    List<ProductionTask> findByWorkstationId(String workstationId);
    ProductionTask save(ProductionTask task);
    void update(ProductionTask task);
    void deleteById(Long id);
}