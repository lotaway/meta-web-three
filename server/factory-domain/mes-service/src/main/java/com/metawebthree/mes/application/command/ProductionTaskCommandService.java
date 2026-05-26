package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldValueRepository;
import com.metawebthree.mes.domain.repository.ProductionTaskRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

@Service
@Transactional
public class ProductionTaskCommandService {
    
    private static final String ENTITY_TYPE = "ProductionTask";
    
    private final ProductionTaskRepository repository;
    private final EntityExtensionFieldValueRepository extensionFieldValueRepository;
    
    public ProductionTaskCommandService(
            ProductionTaskRepository repository,
            EntityExtensionFieldValueRepository extensionFieldValueRepository) {
        this.repository = repository;
        this.extensionFieldValueRepository = extensionFieldValueRepository;
    }
    
    public ProductionTask createTask(String taskNo, Long workOrderId, String workstationId,
                                     String processCode, Integer quantity, String operatorId) {
        if (repository.findByTaskNo(taskNo).isPresent()) {
            throw new IllegalArgumentException("任务编号已存在: " + taskNo);
        }
        
        ProductionTask task = new ProductionTask();
        task.create(taskNo, workOrderId, workstationId, processCode, quantity, operatorId);
        
        return repository.save(task);
    }
    
    public ProductionTask startTask(Long taskId) {
        ProductionTask task = repository.findById(taskId)
                .orElseThrow(() -> new IllegalArgumentException("任务不存在: " + taskId));
        
        task.start();
        repository.update(task);
        
        return task;
    }
    
    public ProductionTask completeTask(Long taskId, Integer qualified, Integer defective) {
        ProductionTask task = repository.findById(taskId)
                .orElseThrow(() -> new IllegalArgumentException("任务不存在: " + taskId));
        
        task.complete(qualified, defective);
        repository.update(task);
        
        return task;
    }
    
    public ProductionTask passQualityCheck(Long taskId) {
        ProductionTask task = repository.findById(taskId)
                .orElseThrow(() -> new IllegalArgumentException("任务不存在: " + taskId));
        
        task.passQualityCheck();
        repository.update(task);
        
        return task;
    }
    
    public ProductionTask failQualityCheck(Long taskId) {
        ProductionTask task = repository.findById(taskId)
                .orElseThrow(() -> new IllegalArgumentException("任务不存在: " + taskId));
        
        task.failQualityCheck();
        repository.update(task);
        
        return task;
    }
    
    public ProductionTask scrapTask(Long taskId) {
        ProductionTask task = repository.findById(taskId)
                .orElseThrow(() -> new IllegalArgumentException("任务不存在: " + taskId));
        
        task.scrap();
        repository.update(task);
        
        return task;
    }
    
    public ProductionTask updateTask(Long taskId, String workstationId, String processCode, 
                                      Integer quantity, String operatorId) {
        ProductionTask task = repository.findById(taskId)
                .orElseThrow(() -> new IllegalArgumentException("任务不存在: " + taskId));
        
        if (workstationId != null) task.setWorkstationId(workstationId);
        if (processCode != null) task.setProcessCode(processCode);
        if (quantity != null) task.setQuantity(quantity);
        if (operatorId != null) task.setOperatorId(operatorId);
        
        repository.update(task);
        
        return task;
    }
    
    public void deleteTask(Long taskId) {
        if (repository.findById(taskId).isEmpty()) {
            throw new IllegalArgumentException("任务不存在: " + taskId);
        }
        repository.deleteById(taskId);
        extensionFieldValueRepository.deleteByEntityTypeAndEntityId(ENTITY_TYPE, taskId);
    }
    
    public void setExtensionFieldValues(Long taskId, Map<String, String> fieldValues) {
        if (repository.findById(taskId).isEmpty()) {
            throw new IllegalArgumentException("任务不存在: " + taskId);
        }
        
        extensionFieldValueRepository.deleteByEntityTypeAndEntityId(ENTITY_TYPE, taskId);
        
        for (Map.Entry<String, String> entry : fieldValues.entrySet()) {
            EntityExtensionFieldValue value = EntityExtensionFieldValue.create(
                    ENTITY_TYPE, taskId, entry.getKey(), entry.getValue());
            extensionFieldValueRepository.save(value);
        }
    }
}