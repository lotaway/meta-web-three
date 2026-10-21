package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.model.OutboxEventDO;
import com.metawebthree.mes.domain.service.MesDomainService;
import com.metawebthree.mes.infrastructure.event.MesEventPublisher;
import com.metawebthree.mes.infrastructure.persistence.mapper.OutboxEventMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.UUID;

@Slf4j
@Service
public class MesCommandService {

    private final MesDomainService domainService;
    private final MesEventPublisher eventPublisher;
    private final OutboxEventMapper outboxEventMapper;

    public MesCommandService(MesDomainService domainService,
                             MesEventPublisher eventPublisher,
                             OutboxEventMapper outboxEventMapper) {
        this.domainService = domainService;
        this.eventPublisher = eventPublisher;
        this.outboxEventMapper = outboxEventMapper;
    }

    @Transactional
    public Long createWorkOrder(String workOrderNo, String productCode, String productName,
                                Integer quantity, String workshopId, String processRouteId) {
        var workOrder = domainService.createWorkOrder(
            workOrderNo, productCode, productName, quantity, workshopId, processRouteId);

        // Persist event to outbox atomically with the domain change
        String eventId = UUID.randomUUID().toString();
        OutboxEventDO outbox = OutboxEventDO.builder()
                .id(workOrder.getId())
                .eventId(eventId)
                .aggregateType("WORK_ORDER")
                .aggregateId(String.valueOf(workOrder.getId()))
                .eventType("WORK_ORDER_CREATED")
                .topic("work.order.created")
                .eventKey(workOrderNo)
                .payload("") // Serialization handled by relay
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(0)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        outboxEventMapper.insert(outbox);

        eventPublisher.publishWorkOrderCreated(workOrder.getId(), workOrderNo, eventId);
        log.info("WorkOrderCreated event persisted to outbox: eventId={}, workOrderId={}", eventId, workOrder.getId());
        return workOrder.getId();
    }

    @Transactional
    public void releaseWorkOrder(Long workOrderId) {
        domainService.releaseWorkOrder(workOrderId);

        String eventId = UUID.randomUUID().toString();
        OutboxEventDO outbox = OutboxEventDO.builder()
                .id(workOrderId)
                .eventId(eventId)
                .aggregateType("WORK_ORDER")
                .aggregateId(String.valueOf(workOrderId))
                .eventType("WORK_ORDER_RELEASED")
                .topic("work.order.released")
                .eventKey(String.valueOf(workOrderId))
                .payload("")
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(0)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        outboxEventMapper.insert(outbox);

        eventPublisher.publishWorkOrderReleased(workOrderId, eventId);
        log.info("WorkOrderReleased event persisted to outbox: eventId={}, workOrderId={}", eventId, workOrderId);
    }

    @Transactional
    public void startWorkOrder(Long workOrderId) {
        domainService.startWorkOrder(workOrderId);

        String eventId = UUID.randomUUID().toString();
        OutboxEventDO outbox = OutboxEventDO.builder()
                .id(workOrderId)
                .eventId(eventId)
                .aggregateType("WORK_ORDER")
                .aggregateId(String.valueOf(workOrderId))
                .eventType("WORK_ORDER_STARTED")
                .topic("work.order.started")
                .eventKey(String.valueOf(workOrderId))
                .payload("")
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(0)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        outboxEventMapper.insert(outbox);

        eventPublisher.publishWorkOrderStarted(workOrderId, eventId);
        log.info("WorkOrderStarted event persisted to outbox: eventId={}, workOrderId={}", eventId, workOrderId);
    }

    @Transactional
    public void completeWorkOrder(Long workOrderId) {
        domainService.completeWorkOrder(workOrderId);

        String eventId = UUID.randomUUID().toString();
        OutboxEventDO outbox = OutboxEventDO.builder()
                .id(workOrderId)
                .eventId(eventId)
                .aggregateType("WORK_ORDER")
                .aggregateId(String.valueOf(workOrderId))
                .eventType("WORK_ORDER_COMPLETED")
                .topic("work.order.completed")
                .eventKey(String.valueOf(workOrderId))
                .payload("")
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(0)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        outboxEventMapper.insert(outbox);

        eventPublisher.publishWorkOrderCompleted(workOrderId, eventId);
        log.info("WorkOrderCompleted event persisted to outbox: eventId={}, workOrderId={}", eventId, workOrderId);
    }

    @Transactional
    public void createTask(String taskNo, Long workOrderId, String workstationId,
                           String processCode, Integer quantity, String operatorId) {
        var task = domainService.createTask(
            taskNo, workOrderId, workstationId, processCode, quantity, operatorId);

        String eventId = UUID.randomUUID().toString();
        OutboxEventDO outbox = OutboxEventDO.builder()
                .id(task.getId())
                .eventId(eventId)
                .aggregateType("TASK")
                .aggregateId(String.valueOf(task.getId()))
                .eventType("TASK_CREATED")
                .topic("task.created")
                .eventKey(taskNo)
                .payload("")
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(0)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        outboxEventMapper.insert(outbox);

        eventPublisher.publishTaskCreated(task.getId(), taskNo, eventId);
        log.info("TaskCreated event persisted to outbox: eventId={}, taskId={}", eventId, task.getId());
    }

    @Transactional
    public void startTask(Long taskId) {
        domainService.startTask(taskId);

        String eventId = UUID.randomUUID().toString();
        OutboxEventDO outbox = OutboxEventDO.builder()
                .id(taskId)
                .eventId(eventId)
                .aggregateType("TASK")
                .aggregateId(String.valueOf(taskId))
                .eventType("TASK_STARTED")
                .topic("task.started")
                .eventKey(String.valueOf(taskId))
                .payload("")
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(0)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        outboxEventMapper.insert(outbox);

        eventPublisher.publishTaskStarted(taskId, eventId);
        log.info("TaskStarted event persisted to outbox: eventId={}, taskId={}", eventId, taskId);
    }

    @Transactional
    public void completeTask(Long taskId, Integer qualified, Integer defective) {
        domainService.completeTask(taskId, qualified, defective);

        String eventId = UUID.randomUUID().toString();
        OutboxEventDO outbox = OutboxEventDO.builder()
                .id(taskId)
                .eventId(eventId)
                .aggregateType("TASK")
                .aggregateId(String.valueOf(taskId))
                .eventType("TASK_COMPLETED")
                .topic("task.completed")
                .eventKey(String.valueOf(taskId))
                .payload("")
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(0)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        outboxEventMapper.insert(outbox);

        eventPublisher.publishTaskCompleted(taskId, qualified, defective, eventId);
        log.info("TaskCompleted event persisted to outbox: eventId={}, taskId={}", eventId, taskId);
    }

    @Transactional
    public void reportEquipmentBreakdown(Long equipmentId) {
        domainService.reportEquipmentBreakdown(equipmentId);

        String eventId = UUID.randomUUID().toString();
        OutboxEventDO outbox = OutboxEventDO.builder()
                .id(equipmentId)
                .eventId(eventId)
                .aggregateType("EQUIPMENT")
                .aggregateId(String.valueOf(equipmentId))
                .eventType("EQUIPMENT_BREAKDOWN")
                .topic("equipment.breakdown")
                .eventKey(String.valueOf(equipmentId))
                .payload("")
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(0)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        outboxEventMapper.insert(outbox);

        eventPublisher.publishEquipmentBreakdown(equipmentId, eventId);
        log.info("EquipmentBreakdown event persisted to outbox: eventId={}, equipmentId={}", eventId, equipmentId);
    }

    @Transactional
    public void repairEquipment(Long equipmentId) {
        domainService.repairEquipment(equipmentId);

        String eventId = UUID.randomUUID().toString();
        OutboxEventDO outbox = OutboxEventDO.builder()
                .id(equipmentId)
                .eventId(eventId)
                .aggregateType("EQUIPMENT")
                .aggregateId(String.valueOf(equipmentId))
                .eventType("EQUIPMENT_REPAIRED")
                .topic("equipment.repaired")
                .eventKey(String.valueOf(equipmentId))
                .payload("")
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(0)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
        outboxEventMapper.insert(outbox);

        eventPublisher.publishEquipmentRepaired(equipmentId, eventId);
        log.info("EquipmentRepaired event persisted to outbox: eventId={}, equipmentId={}", eventId, equipmentId);
    }
}