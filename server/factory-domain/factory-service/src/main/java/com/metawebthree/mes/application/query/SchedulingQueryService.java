package com.metawebthree.mes.application.query;

import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResource;
import com.metawebthree.mes.domain.repository.scheduling.ScheduleOrderRepository;
import com.metawebthree.mes.domain.repository.scheduling.ScheduleResourceRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
public class SchedulingQueryService {

    private final ScheduleOrderRepository scheduleOrderRepository;
    private final ScheduleResourceRepository scheduleResourceRepository;

    public SchedulingQueryService(ScheduleOrderRepository scheduleOrderRepository,
                                  ScheduleResourceRepository scheduleResourceRepository) {
        this.scheduleOrderRepository = scheduleOrderRepository;
        this.scheduleResourceRepository = scheduleResourceRepository;
    }

    public Optional<ScheduleOrder> findOrderById(Long id) {
        return scheduleOrderRepository.findById(id);
    }

    public Optional<ScheduleOrder> findOrderByScheduleNo(String scheduleNo) {
        return scheduleOrderRepository.findByScheduleNo(scheduleNo);
    }

    public List<ScheduleOrder> findOrdersByStatus(String status) {
        return scheduleOrderRepository.findByStatus(ScheduleOrder.ScheduleStatus.valueOf(status));
    }

    public List<ScheduleOrder> findOrdersByWorkshop(String workshopId) {
        return scheduleOrderRepository.findByWorkshopId(workshopId);
    }

    public List<ScheduleOrder> findOrdersByDueDateRange(LocalDateTime start, LocalDateTime end) {
        return scheduleOrderRepository.findByDueDateBetween(start, end);
    }

    public List<ScheduleOrder> findOverdueOrders() {
        return scheduleOrderRepository.findOverdueOrders();
    }

    public List<ScheduleOrder> findAllOrders() {
        return scheduleOrderRepository.findAll();
    }

    public Optional<ScheduleResource> findResourceById(Long id) {
        return scheduleResourceRepository.findById(id);
    }

    public Optional<ScheduleResource> findResourceByCode(String resourceCode) {
        return scheduleResourceRepository.findByResourceCode(resourceCode);
    }

    public List<ScheduleResource> findResourcesByWorkshop(String workshopId) {
        return scheduleResourceRepository.findByWorkshopId(workshopId);
    }

    public List<ScheduleResource> findResourcesByType(String resourceType) {
        return scheduleResourceRepository.findByResourceType(ScheduleResource.ResourceType.valueOf(resourceType));
    }

    public List<ScheduleResource> findAllResources() {
        return scheduleResourceRepository.findAll();
    }
}
