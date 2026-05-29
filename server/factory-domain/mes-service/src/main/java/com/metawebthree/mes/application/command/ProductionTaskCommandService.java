package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import com.metawebthree.mes.domain.entity.PokaYokeRule;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.entity.WorkReport;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldValueRepository;
import com.metawebthree.mes.domain.repository.ProductionTaskRepository;
import com.metawebthree.mes.domain.repository.WorkReportRepository;
import com.metawebthree.mes.domain.service.PokaYokeService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
@Transactional
public class ProductionTaskCommandService {
    
    private static final String ENTITY_TYPE = "ProductionTask";
    
    private final ProductionTaskRepository repository;
    private final EntityExtensionFieldValueRepository extensionFieldValueRepository;
    private final WorkReportRepository workReportRepository;
    private final PokaYokeService pokaYokeService;
    
    public ProductionTaskCommandService(
            ProductionTaskRepository repository,
            EntityExtensionFieldValueRepository extensionFieldValueRepository,
            WorkReportRepository workReportRepository,
            PokaYokeService pokaYokeService) {
        this.repository = repository;
        this.extensionFieldValueRepository = extensionFieldValueRepository;
        this.workReportRepository = workReportRepository;
        this.pokaYokeService = pokaYokeService;
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
    
    public ProductionTask completeTask(Long taskId, Integer qualified, Integer defective, 
                                        Integer durationMinutes, String parameterValuesJson) {
        ProductionTask task = repository.findById(taskId)
                .orElseThrow(() -> new IllegalArgumentException("任务不存在: " + taskId));
        
        if (parameterValuesJson != null && !parameterValuesJson.isEmpty()) {
            validateParameterValues(parameterValuesJson, task);
        }
        
        List<PokaYokeService.PokaYokeResult> pokayokeResults = performPokayokeCheck(task);
        
        boolean hasBlockingError = pokayokeResults.stream()
            .anyMatch(r -> r.actionType() == PokaYokeRule.CheckAction.ActionType.BLOCK && !r.passed());
        
        if (hasBlockingError) {
            StringBuilder errorMsg = new StringBuilder("防错检查未通过: ");
            pokayokeResults.stream()
                .filter(r -> r.actionType() == PokaYokeRule.CheckAction.ActionType.BLOCK && !r.passed())
                .forEach(r -> errorMsg.append(r.message()).append("; "));
            throw new IllegalArgumentException(errorMsg.toString());
        }
        
        task.complete(qualified, defective, durationMinutes);
        repository.update(task);
        
        createWorkReport(task, qualified, defective, durationMinutes, parameterValuesJson);
        
        return task;
    }
    
    public ProductionTask completeTask(Long taskId, Integer qualified, Integer defective) {
        return completeTask(taskId, qualified, defective, null, null);
    }
    
    private void validateParameterValues(String parameterValuesJson, ProductionTask task) {
    }
    
    private List<PokaYokeService.PokaYokeResult> performPokayokeCheck(ProductionTask task) {
        return List.of();
    }
    
    private void createWorkReport(ProductionTask task, Integer qualified, Integer defective,
                                   Integer durationMinutes, String parameterValuesJson) {
        WorkReport report = new WorkReport();
        String reportNo = generateReportNo();
        
        report.create(
            reportNo,
            task.getId(),
            task.getTaskNo(),
            task.getWorkOrderId(),
            null,
            task.getWorkstationId(),
            null,
            task.getProcessCode(),
            null,
            null,
            task.getOperatorId(),
            null
        );
        
        report.recordOutput(
            (qualified != null ? qualified : 0) + (defective != null ? defective : 0),
            qualified != null ? qualified : 0,
            defective != null ? defective : 0,
            durationMinutes != null ? durationMinutes : 0
        );
        
        if (parameterValuesJson != null) {
            report.setParameterValues(parameterValuesJson);
        }
        
        report.submit();
        
        workReportRepository.save(report);
    }
    
    private String generateReportNo() {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
        String uuid = UUID.randomUUID().toString().substring(0, 8).toUpperCase();
        return "WR" + timestamp + uuid;
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