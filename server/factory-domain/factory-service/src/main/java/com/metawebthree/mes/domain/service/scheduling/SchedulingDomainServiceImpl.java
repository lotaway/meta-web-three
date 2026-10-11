package com.metawebthree.mes.domain.service.scheduling;

import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder.ScheduleOperation;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder.ScheduleOperationStatus;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResource;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResource.ResourceStatus;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResult;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResult.ScheduleConflict;
import com.metawebthree.mes.domain.repository.scheduling.ScheduleOrderRepository;
import com.metawebthree.mes.domain.repository.scheduling.ScheduleResourceRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
public class SchedulingDomainServiceImpl implements SchedulingDomainService {

    private final ScheduleOrderRepository scheduleOrderRepository;
    private final ScheduleResourceRepository scheduleResourceRepository;

    public SchedulingDomainServiceImpl(
            ScheduleOrderRepository scheduleOrderRepository,
            ScheduleResourceRepository scheduleResourceRepository) {
        this.scheduleOrderRepository = scheduleOrderRepository;
        this.scheduleResourceRepository = scheduleResourceRepository;
    }

    @Override
    public ScheduleResult forwardSchedule(List<ScheduleOrder> orders, String workshopId) {
        long startTime = System.currentTimeMillis();
        List<ScheduleOrder> scheduled = new ArrayList<>();
        List<ScheduleConflict> conflicts = new ArrayList<>();
        List<ScheduleResource> resources = scheduleResourceRepository.findByWorkshopId(workshopId);

        List<ScheduleOrder> sortedOrders = sortByPriorityAndDueDate(orders);

        for (ScheduleOrder order : sortedOrders) {
            try {
                List<ScheduleOperation> ops = order.getOperations();
                if (ops == null || ops.isEmpty()) {
                    conflicts.add(new ScheduleConflict(order.getOrderNo(), order.getProductCode(),
                        "N/A", "NO_OPERATIONS", "Order has no defined operations"));
                    continue;
                }
                LocalDateTime currentTime = LocalDateTime.now();
                for (ScheduleOperation op : ops) {
                    ScheduleResource resource = findAvailableResource(resources, op.getResourceCode(), currentTime);
                    if (resource == null) {
                        conflicts.add(new ScheduleConflict(order.getOrderNo(), order.getProductCode(),
                            op.getResourceCode(), "RESOURCE_UNAVAILABLE",
                            "No available resource for operation " + op.getOperationCode()));
                        break;
                    }
                    BigDecimal totalMinutes = op.getSetupTimeMinutes()
                        .add(op.getProcessingTimeMinutes())
                        .add(op.getTeardownTimeMinutes() != null ? op.getTeardownTimeMinutes() : BigDecimal.ZERO);
                    LocalDateTime opEnd = currentTime.plusMinutes(totalMinutes.longValue());
                    op.setScheduledStartTime(currentTime);
                    op.setScheduledEndTime(opEnd);
                    op.setStatus(ScheduleOperationStatus.SCHEDULED);
                    resource.occupy(currentTime, opEnd, order.getId(), order.getScheduleNo());
                    currentTime = opEnd;
                }
                if (ops.stream().allMatch(op -> op.getStatus() == ScheduleOperationStatus.SCHEDULED)) {
                    LocalDateTime start = ops.get(0).getScheduledStartTime();
                    LocalDateTime end = ops.get(ops.size() - 1).getScheduledEndTime();
                    order.schedule(start, end, ops);
                    scheduleOrderRepository.update(order);
                    scheduled.add(order);
                }
            } catch (Exception e) {
                log.error("Failed to schedule order {}: {}", order.getOrderNo(), e);
                conflicts.add(new ScheduleConflict(order.getOrderNo(), order.getProductCode(),
                    "N/A", "SCHEDULING_ERROR", e.getMessage()));
            }
        }

        long elapsed = System.currentTimeMillis() - startTime;
        if (conflicts.isEmpty()) {
            return ScheduleResult.success(scheduled, ScheduleResult.ScheduleDirection.FORWARD, elapsed);
        } else if (!scheduled.isEmpty()) {
            return ScheduleResult.partial(scheduled, conflicts, ScheduleResult.ScheduleDirection.FORWARD, elapsed);
        }
        return ScheduleResult.failed(conflicts, elapsed);
    }

    @Override
    public ScheduleResult backwardSchedule(List<ScheduleOrder> orders, String workshopId) {
        long startTime = System.currentTimeMillis();
        List<ScheduleOrder> scheduled = new ArrayList<>();
        List<ScheduleConflict> conflicts = new ArrayList<>();
        List<ScheduleResource> resources = scheduleResourceRepository.findByWorkshopId(workshopId);

        List<ScheduleOrder> sortedOrders = sortByPriorityAndDueDate(orders);

        for (ScheduleOrder order : sortedOrders) {
            try {
                List<ScheduleOperation> ops = order.getOperations();
                if (ops == null || ops.isEmpty()) {
                    conflicts.add(new ScheduleConflict(order.getOrderNo(), order.getProductCode(),
                        "N/A", "NO_OPERATIONS", "Order has no defined operations"));
                    continue;
                }
                List<ScheduleOperation> reversedOps = new ArrayList<>(ops);
                Collections.reverse(reversedOps);
                LocalDateTime currentTime = order.getDueDate() != null ? order.getDueDate() : LocalDateTime.now().plusDays(7);

                for (ScheduleOperation op : reversedOps) {
                    BigDecimal totalMinutes = op.getSetupTimeMinutes()
                        .add(op.getProcessingTimeMinutes())
                        .add(op.getTeardownTimeMinutes() != null ? op.getTeardownTimeMinutes() : BigDecimal.ZERO);
                    LocalDateTime opStart = currentTime.minusMinutes(totalMinutes.longValue());

                    ScheduleResource resource = findAvailableResource(resources, op.getResourceCode(), opStart);
                    if (resource == null) {
                        conflicts.add(new ScheduleConflict(order.getOrderNo(), order.getProductCode(),
                            op.getResourceCode(), "RESOURCE_UNAVAILABLE",
                            "No available resource for operation " + op.getOperationCode()));
                        break;
                    }
                    op.setScheduledStartTime(opStart);
                    op.setScheduledEndTime(currentTime);
                    op.setStatus(ScheduleOperationStatus.SCHEDULED);
                    resource.occupy(opStart, currentTime, order.getId(), order.getScheduleNo());
                    currentTime = opStart;
                }
                if (ops.stream().allMatch(op -> op.getStatus() == ScheduleOperationStatus.SCHEDULED)) {
                    LocalDateTime start = ops.get(0).getScheduledStartTime();
                    LocalDateTime end = ops.get(ops.size() - 1).getScheduledEndTime();
                    order.schedule(start, end, ops);
                    scheduleOrderRepository.update(order);
                    scheduled.add(order);
                }
            } catch (Exception e) {
                log.error("Failed to backward schedule order {}: {}", order.getOrderNo(), e);
                conflicts.add(new ScheduleConflict(order.getOrderNo(), order.getProductCode(),
                    "N/A", "SCHEDULING_ERROR", e.getMessage()));
            }
        }

        long elapsed = System.currentTimeMillis() - startTime;
        if (conflicts.isEmpty()) {
            return ScheduleResult.success(scheduled, ScheduleResult.ScheduleDirection.BACKWARD, elapsed);
        } else if (!scheduled.isEmpty()) {
            return ScheduleResult.partial(scheduled, conflicts, ScheduleResult.ScheduleDirection.BACKWARD, elapsed);
        }
        return ScheduleResult.failed(conflicts, elapsed);
    }

    @Override
    public ScheduleResult reschedule(Long orderId) {
        ScheduleOrder order = scheduleOrderRepository.findById(orderId)
            .orElseThrow(() -> new IllegalArgumentException("Schedule order not found: " + orderId));
        order.cancel();
        scheduleOrderRepository.update(order);
        return forwardSchedule(List.of(order), order.getWorkshopId());
    }

    @Override
    public void releaseResource(Long resourceId, LocalDateTime endTime) {
        scheduleResourceRepository.findById(resourceId).ifPresent(resource -> {
            resource.release(endTime);
            scheduleResourceRepository.update(resource);
        });
    }

    private List<ScheduleOrder> sortByPriorityAndDueDate(List<ScheduleOrder> orders) {
        return orders.stream()
            .sorted(Comparator
                .comparingInt((ScheduleOrder o) -> o.getPriority() != null ? o.getPriority().ordinal() : 2)
                .thenComparing(o -> o.getDueDate() != null ? o.getDueDate() : LocalDateTime.MAX))
            .collect(Collectors.toList());
    }

    private ScheduleResource findAvailableResource(List<ScheduleResource> resources, String resourceCode, LocalDateTime time) {
        return resources.stream()
            .filter(r -> r.getResourceCode().equals(resourceCode))
            .filter(r -> r.getStatus() == ResourceStatus.AVAILABLE || r.getStatus() == ResourceStatus.OCCUPIED)
            .filter(r -> r.isAvailable(time, time.plusHours(8)))
            .findFirst()
            .orElseGet(() -> resources.stream()
                .filter(r -> r.getResourceCode().equals(resourceCode))
                .filter(r -> r.getStatus() != ResourceStatus.OFFLINE)
                .findFirst()
                .orElse(null));
    }
}
