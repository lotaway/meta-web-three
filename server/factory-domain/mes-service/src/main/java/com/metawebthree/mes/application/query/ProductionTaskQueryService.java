package com.metawebthree.mes.application.query;

import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldValueRepository;
import com.metawebthree.mes.domain.repository.ProductionTaskRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class ProductionTaskQueryService {
    
    private static final String ENTITY_TYPE = "ProductionTask";
    
    private final ProductionTaskRepository repository;
    private final EntityExtensionFieldValueRepository extensionFieldValueRepository;
    
    public ProductionTaskQueryService(
            ProductionTaskRepository repository,
            EntityExtensionFieldValueRepository extensionFieldValueRepository) {
        this.repository = repository;
        this.extensionFieldValueRepository = extensionFieldValueRepository;
    }
    
    public Optional<ProductionTask> findById(Long id) {
        return repository.findById(id);
    }
    
    public Optional<ProductionTask> findByTaskNo(String taskNo) {
        return repository.findByTaskNo(taskNo);
    }
    
    public List<ProductionTask> findByWorkOrderId(Long workOrderId) {
        return repository.findByWorkOrderId(workOrderId);
    }
    
    public List<ProductionTask> findByStatus(ProductionTask.TaskStatus status) {
        return repository.findByStatus(status);
    }
    
    public List<ProductionTask> findByWorkstationId(String workstationId) {
        return repository.findByWorkstationId(workstationId);
    }
    
    public List<ProductionTask> findAll() {
        return repository.findAll();
    }
    
    public List<EntityExtensionFieldValue> getExtensionFieldValues(Long taskId) {
        return extensionFieldValueRepository.findByEntityTypeAndEntityId(ENTITY_TYPE, taskId);
    }
    
    public long countByWorkOrderId(Long workOrderId) {
        return repository.findByWorkOrderId(workOrderId).size();
    }
    
    public long countByStatus(ProductionTask.TaskStatus status) {
        return repository.findByStatus(status).size();
    }
}