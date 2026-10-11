package com.metawebthree.mes.domain.repository.scheduling;

import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface ScheduleOrderRepository {
    Optional<ScheduleOrder> findById(Long id);
    Optional<ScheduleOrder> findByScheduleNo(String scheduleNo);
    List<ScheduleOrder> findByStatus(ScheduleOrder.ScheduleStatus status);
    List<ScheduleOrder> findByWorkshopId(String workshopId);
    List<ScheduleOrder> findByDueDateBetween(LocalDateTime start, LocalDateTime end);
    List<ScheduleOrder> findOverdueOrders();
    List<ScheduleOrder> findAll();
    ScheduleOrder save(ScheduleOrder scheduleOrder);
    void update(ScheduleOrder scheduleOrder);
    void deleteById(Long id);
}
