package com.metawebthree.production.domain.repository;

import com.metawebthree.production.domain.entity.ProductionSchedule;
import java.util.List;
import java.util.Optional;

public interface ProductionScheduleRepository {
    Optional<ProductionSchedule> findById(Long id);
    Optional<ProductionSchedule> findByScheduleCode(String scheduleCode);
    List<ProductionSchedule> findByOrderCode(String orderCode);
    List<ProductionSchedule> findByStationCode(String stationCode);
    List<ProductionSchedule> findByStatus(ProductionSchedule.ScheduleStatus status);
    List<ProductionSchedule> findAll();
    ProductionSchedule save(ProductionSchedule schedule);
    void delete(ProductionSchedule schedule);
}