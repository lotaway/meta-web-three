package com.metawebthree.routeoptimizer.infrastructure.persistence.repository;

import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface RoutePlanJpaRepository extends JpaRepository<RoutePlan, Long> {
    Optional<RoutePlan> findByPlanCode(String planCode);
    List<RoutePlan> findByStatus(RoutePlan.RouteStatus status);
    List<RoutePlan> findByVehicleCode(String vehicleCode);
    List<RoutePlan> findByPlannedStartTimeBetween(LocalDateTime start, LocalDateTime end);
}
