package com.metawebthree.routeoptimizer.domain.repository;

import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import java.util.List;
import java.util.Optional;

public interface RoutePlanRepository {
    Optional<RoutePlan> findById(Long id);
    Optional<RoutePlan> findByPlanCode(String planCode);
    List<RoutePlan> findByStatus(RoutePlan.RouteStatus status);
    List<RoutePlan> findByVehicleCode(String vehicleCode);
    List<RoutePlan> findAll();
    void save(RoutePlan routePlan);
    void delete(RoutePlan routePlan);
    List<RoutePlan> findByPlannedStartTimeBetween(java.time.LocalDateTime start, java.time.LocalDateTime end);
}