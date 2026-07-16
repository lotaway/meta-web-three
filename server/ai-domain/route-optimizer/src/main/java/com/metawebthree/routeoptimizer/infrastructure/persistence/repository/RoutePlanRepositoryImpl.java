package com.metawebthree.routeoptimizer.infrastructure.persistence.repository;

import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import com.metawebthree.routeoptimizer.domain.repository.RoutePlanRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class RoutePlanRepositoryImpl implements RoutePlanRepository {
    private final RoutePlanJpaRepository repository;

    public RoutePlanRepositoryImpl(RoutePlanJpaRepository repository) {
        this.repository = repository;
    }

    @Override
    public Optional<RoutePlan> findById(Long id) {
        return repository.findById(id);
    }

    @Override
    public Optional<RoutePlan> findByPlanCode(String planCode) {
        return repository.findByPlanCode(planCode);
    }

    @Override
    public List<RoutePlan> findByStatus(RoutePlan.RouteStatus status) {
        return repository.findByStatus(status);
    }

    @Override
    public List<RoutePlan> findByVehicleCode(String vehicleCode) {
        return repository.findByVehicleCode(vehicleCode);
    }

    @Override
    public List<RoutePlan> findAll() {
        return repository.findAll();
    }

    @Override
    public RoutePlan save(RoutePlan routePlan) {
        return repository.save(routePlan);
    }

    @Override
    public void delete(RoutePlan routePlan) {
        repository.delete(routePlan);
    }

    @Override
    public List<RoutePlan> findByPlannedStartTimeBetween(LocalDateTime start, LocalDateTime end) {
        return repository.findByPlannedStartTimeBetween(start, end);
    }
}
