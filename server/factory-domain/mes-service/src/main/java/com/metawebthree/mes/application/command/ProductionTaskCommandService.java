package com.metawebthree.mes.application.command;

import com.metawebthree.mes.application.event.CrossDomainEventPublisher;
import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import com.metawebthree.mes.domain.entity.PokaYokeRule;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.WorkReport;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldValueRepository;
import com.metawebthree.mes.domain.repository.ProductionTaskRepository;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
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
    private static final String REPORT_NO_FORMAT = "yyyyMMddHHmmss";
    private static final String REPORT_NO_PREFIX = "WR";
    private static final int UUID_TRUNCATE_LENGTH = 8;

    private final ProductionTaskRepository repository;
    private final EntityExtensionFieldValueRepository extensionFieldValueRepository;
    private final WorkReportRepository workReportRepository;
    private final PokaYokeService pokaYokeService;
    private final CrossDomainEventPublisher crossDomainEventPublisher;
    private final WorkOrderRepository workOrderRepository;

    public ProductionTaskCommandService(
            ProductionTaskRepository repository,
            EntityExtensionFieldValueRepository extensionFieldValueRepository,
            WorkReportRepository workReportRepository,
            PokaYokeService pokaYokeService,
            CrossDomainEventPublisher crossDomainEventPublisher,
            WorkOrderRepository workOrderRepository) {
        this.repository = repository;
        this.extensionFieldValueRepository = extensionFieldValueRepository;
        this.workReportRepository = workReportRepository;
        this.pokaYokeService = pokaYokeService;
        this.crossDomainEventPublisher = crossDomainEventPublisher;
        this.workOrderRepository = workOrderRepository;
    }

    public ProductionTask prepareCreateTask(String taskNo, Long workOrderId, Long workstationId,
                                            String processCode, Integer quantity, String operatorId) {
        if (repository.findByTaskNo(taskNo).isPresent()) {
            throw new IllegalArgumentException("任务编号已存在: " + taskNo);
        }
        ProductionTask task = new ProductionTask();
        task.create(taskNo, workOrderId, workstationId, processCode, quantity, operatorId);
        return task;
    }

    public void saveTask(ProductionTask task) {
        repository.save(task);
    }

    public ProductionTask prepareStartTask(Long taskId) {
        ProductionTask task = findTaskOrThrow(taskId);
        task.start();
        return task;
    }

    public void saveStartTask(ProductionTask task) {
        repository.update(task);
    }

    public ProductionTask prepareCompleteTask(Long taskId, Integer qualified, Integer defective,
                                              Integer durationMinutes, String parameterValuesJson) {
        ProductionTask task = findTaskOrThrow(taskId);
        validateParameters(parameterValuesJson);
        checkPokayoke(task);
        task.complete(qualified, defective, durationMinutes);
        return task;
    }

    public ProductionTask prepareCompleteTask(Long taskId, Integer qualified, Integer defective) {
        return prepareCompleteTask(taskId, qualified, defective, null, null);
    }

    public void saveCompleteTask(ProductionTask task, Integer qualified, Integer defective,
                                 Integer durationMinutes, String parameterValuesJson) {
        repository.update(task);
        createWorkReport(task, qualified, defective, durationMinutes, parameterValuesJson);
        publishTaskCompletedEvent(task, qualified, defective);
    }

    public void saveCompleteTask(ProductionTask task, Integer qualified, Integer defective) {
        saveCompleteTask(task, qualified, defective, null, null);
    }

    private ProductionTask findTaskOrThrow(Long taskId) {
        return repository.findById(taskId)
                .orElseThrow(() -> new IllegalArgumentException("任务不存在: " + taskId));
    }

    private void validateParameters(String parameterValuesJson) {
    }

    private void checkPokayoke(ProductionTask task) {
        List<PokaYokeService.PokaYokeResult> results = performPokayokeCheck(task);
        boolean hasBlockingError = results.stream()
            .anyMatch(r -> r.actionType() == PokaYokeRule.CheckAction.ActionType.BLOCK && !r.passed());
        if (hasBlockingError) {
            StringBuilder errorMsg = new StringBuilder("防错检查未通过: ");
            results.stream()
                .filter(r -> r.actionType() == PokaYokeRule.CheckAction.ActionType.BLOCK && !r.passed())
                .forEach(r -> errorMsg.append(r.message()).append("; "));
            throw new IllegalArgumentException(errorMsg.toString());
        }
    }

    private List<PokaYokeService.PokaYokeResult> performPokayokeCheck(ProductionTask task) {
        return List.of();
    }

    private void publishTaskCompletedEvent(ProductionTask task, Integer qualified, Integer defective) {
        workOrderRepository.findById(task.getWorkOrderId()).ifPresent(workOrder ->
            crossDomainEventPublisher.publishTaskCompleted(
                task.getId(), task.getTaskNo(), workOrder.getId(),
                workOrder.getWorkOrderNo(), workOrder.getProductCode(),
                qualified, defective)
        );
    }

    private void createWorkReport(ProductionTask task, Integer qualified, Integer defective,
                                   Integer durationMinutes, String parameterValuesJson) {
        WorkReport report = buildWorkReport(task, qualified, defective, durationMinutes);
        if (parameterValuesJson != null) {
            report.setParameterValues(parameterValuesJson);
        }
        report.submit();
        workReportRepository.save(report);
    }

    private WorkReport buildWorkReport(ProductionTask task, Integer qualified, Integer defective,
                                        Integer durationMinutes) {
        WorkReport report = buildReportData(task);
        recordOutput(report, qualified, defective, durationMinutes);
        return report;
    }

    private WorkReport buildReportData(ProductionTask task) {
        WorkReport report = new WorkReport();
        report.create(
            generateReportNo(),
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
        return report;
    }

    private void recordOutput(WorkReport report, Integer qualified, Integer defective, Integer durationMinutes) {
        report.recordOutput(
            (qualified != null ? qualified : 0) + (defective != null ? defective : 0),
            qualified != null ? qualified : 0,
            defective != null ? defective : 0,
            durationMinutes != null ? durationMinutes : 0
        );
    }

    private String generateReportNo() {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern(REPORT_NO_FORMAT));
        String uuid = UUID.randomUUID().toString().substring(0, UUID_TRUNCATE_LENGTH).toUpperCase();
        return REPORT_NO_PREFIX + timestamp + uuid;
    }

    public ProductionTask preparePassQualityCheck(Long taskId) {
        ProductionTask task = findTaskOrThrow(taskId);
        task.passQualityCheck();
        return task;
    }

    public void savePassQualityCheck(ProductionTask task) {
        repository.update(task);
    }

    public ProductionTask prepareFailQualityCheck(Long taskId) {
        ProductionTask task = findTaskOrThrow(taskId);
        task.failQualityCheck();
        return task;
    }

    public void saveFailQualityCheck(ProductionTask task) {
        repository.update(task);
    }

    public ProductionTask prepareScrapTask(Long taskId) {
        ProductionTask task = findTaskOrThrow(taskId);
        task.scrap();
        return task;
    }

    public void saveScrapTask(ProductionTask task) {
        repository.update(task);
    }

    public ProductionTask prepareUpdateTask(Long taskId, Long workstationId, String processCode,
                                            Integer quantity, String operatorId) {
        ProductionTask task = findTaskOrThrow(taskId);
        if (workstationId != null) task.setWorkstationId(workstationId);
        if (processCode != null) task.setProcessCode(processCode);
        if (quantity != null) task.setQuantity(quantity);
        if (operatorId != null) task.setOperatorId(operatorId);
        return task;
    }

    public void saveUpdateTask(ProductionTask task) {
        repository.update(task);
    }

    public void deleteTask(Long taskId) {
        repository.deleteById(taskId);
        extensionFieldValueRepository.deleteByEntityTypeAndEntityId(ENTITY_TYPE, taskId);
    }

    public void setExtensionFieldValues(Long taskId, Map<String, String> fieldValues) {
        extensionFieldValueRepository.deleteByEntityTypeAndEntityId(ENTITY_TYPE, taskId);
        for (Map.Entry<String, String> entry : fieldValues.entrySet()) {
            EntityExtensionFieldValue value = EntityExtensionFieldValue.create(
                    ENTITY_TYPE, taskId, entry.getKey(), entry.getValue());
            extensionFieldValueRepository.save(value);
        }
    }
}
