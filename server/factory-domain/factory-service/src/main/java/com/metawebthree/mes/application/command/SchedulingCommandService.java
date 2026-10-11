package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder.Priority;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder.ScheduleOperation;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder.ScheduleOperationStatus;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResource;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResult;
import com.metawebthree.mes.domain.repository.scheduling.ScheduleOrderRepository;
import com.metawebthree.mes.domain.repository.scheduling.ScheduleResourceRepository;
import com.metawebthree.mes.domain.service.scheduling.SchedulingDomainService;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Service
public class SchedulingCommandService {

    private final ScheduleOrderRepository scheduleOrderRepository;
    private final ScheduleResourceRepository scheduleResourceRepository;
    private final SchedulingDomainService schedulingDomainService;

    public SchedulingCommandService(ScheduleOrderRepository scheduleOrderRepository,
                                    ScheduleResourceRepository scheduleResourceRepository,
                                    SchedulingDomainService schedulingDomainService) {
        this.scheduleOrderRepository = scheduleOrderRepository;
        this.scheduleResourceRepository = scheduleResourceRepository;
        this.schedulingDomainService = schedulingDomainService;
    }

    public ScheduleOrder createScheduleOrder(String scheduleNo, String orderNo, String productCode,
                                              String productName, BigDecimal quantity, LocalDateTime dueDate,
                                              String priority, String workshopId, String routeCode) {
        ScheduleOrder entity = new ScheduleOrder();
        entity.create(scheduleNo, orderNo, productCode, productName, quantity, dueDate,
            Priority.valueOf(priority), workshopId, routeCode);
        return scheduleOrderRepository.save(entity);
    }

    public ScheduleOrder addOperations(Long orderId, List<OperationRequest> operations) {
        ScheduleOrder order = scheduleOrderRepository.findById(orderId)
            .orElseThrow(() -> new IllegalArgumentException("Schedule order not found: " + orderId));
        List<ScheduleOperation> ops = new ArrayList<>();
        int seq = 1;
        for (OperationRequest req : operations) {
            ScheduleOperation op = new ScheduleOperation();
            op.setOperationCode(req.getOperationCode());
            op.setOperationName(req.getOperationName());
            op.setSequenceNo(seq++);
            op.setResourceCode(req.getResourceCode());
            op.setResourceName(req.getResourceName());
            op.setSetupTimeMinutes(req.getSetupTimeMinutes());
            op.setProcessingTimeMinutes(req.getProcessingTimeMinutes());
            op.setTeardownTimeMinutes(req.getTeardownTimeMinutes());
            op.setStatus(ScheduleOperationStatus.PENDING);
            ops.add(op);
        }
        order.setOperations(ops);
        scheduleOrderRepository.update(order);
        return order;
    }

    public ScheduleResult scheduleForward(String workshopId) {
        List<ScheduleOrder> orders = scheduleOrderRepository.findByWorkshopId(workshopId);
        return schedulingDomainService.forwardSchedule(orders, workshopId);
    }

    public ScheduleResult scheduleBackward(String workshopId) {
        List<ScheduleOrder> orders = scheduleOrderRepository.findByWorkshopId(workshopId);
        return schedulingDomainService.backwardSchedule(orders, workshopId);
    }

    public ScheduleResult reschedule(Long orderId) {
        return schedulingDomainService.reschedule(orderId);
    }

    public ScheduleOrder startOrder(Long id) {
        ScheduleOrder order = scheduleOrderRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Schedule order not found: " + id));
        order.start();
        scheduleOrderRepository.update(order);
        return order;
    }

    public ScheduleOrder completeOrder(Long id) {
        ScheduleOrder order = scheduleOrderRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Schedule order not found: " + id));
        order.complete();
        scheduleOrderRepository.update(order);
        return order;
    }

    public ScheduleOrder cancelOrder(Long id) {
        ScheduleOrder order = scheduleOrderRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Schedule order not found: " + id));
        order.cancel();
        scheduleOrderRepository.update(order);
        return order;
    }

    public ScheduleOrder markDelayed(Long id) {
        ScheduleOrder order = scheduleOrderRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Schedule order not found: " + id));
        order.markDelayed();
        scheduleOrderRepository.update(order);
        return order;
    }

    public ScheduleOrder updateProgress(Long id, BigDecimal completedQty) {
        ScheduleOrder order = scheduleOrderRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Schedule order not found: " + id));
        order.updateCompletedQuantity(completedQty);
        scheduleOrderRepository.update(order);
        return order;
    }

    public void deleteOrder(Long id) {
        scheduleOrderRepository.deleteById(id);
    }

    public ScheduleResource createResource(String resourceCode, String resourceName, String resourceType,
                                            String workshopId) {
        ScheduleResource entity = new ScheduleResource();
        entity.create(resourceCode, resourceName,
            ScheduleResource.ResourceType.valueOf(resourceType), workshopId);
        return scheduleResourceRepository.save(entity);
    }

    public ScheduleResource updateResource(Long id, String resourceName, Double capacityPerShift,
                                            String status, String description) {
        ScheduleResource resource = scheduleResourceRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Schedule resource not found: " + id));
        resource.setResourceName(resourceName);
        resource.setCapacityPerShift(capacityPerShift);
        if (status != null) {
            resource.setStatus(ScheduleResource.ResourceStatus.valueOf(status));
        }
        resource.setDescription(description);
        scheduleResourceRepository.update(resource);
        return resource;
    }

    public void deleteResource(Long id) {
        scheduleResourceRepository.deleteById(id);
    }

    public static class OperationRequest {
        private String operationCode;
        private String operationName;
        private String resourceCode;
        private String resourceName;
        private BigDecimal setupTimeMinutes;
        private BigDecimal processingTimeMinutes;
        private BigDecimal teardownTimeMinutes;

        public String getOperationCode() { return operationCode; }
        public void setOperationCode(String operationCode) { this.operationCode = operationCode; }
        public String getOperationName() { return operationName; }
        public void setOperationName(String operationName) { this.operationName = operationName; }
        public String getResourceCode() { return resourceCode; }
        public void setResourceCode(String resourceCode) { this.resourceCode = resourceCode; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public BigDecimal getSetupTimeMinutes() { return setupTimeMinutes; }
        public void setSetupTimeMinutes(BigDecimal setupTimeMinutes) { this.setupTimeMinutes = setupTimeMinutes; }
        public BigDecimal getProcessingTimeMinutes() { return processingTimeMinutes; }
        public void setProcessingTimeMinutes(BigDecimal processingTimeMinutes) { this.processingTimeMinutes = processingTimeMinutes; }
        public BigDecimal getTeardownTimeMinutes() { return teardownTimeMinutes; }
        public void setTeardownTimeMinutes(BigDecimal teardownTimeMinutes) { this.teardownTimeMinutes = teardownTimeMinutes; }
    }
}
