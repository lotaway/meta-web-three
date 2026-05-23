package com.metawebthree.routeoptimizer.infrastructure.persistence.repository;

import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import com.metawebthree.routeoptimizer.domain.repository.RoutePlanRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Repository
public class RoutePlanRepositoryImpl implements RoutePlanRepository {
    private final Map<Long, RoutePlan> storage = new ConcurrentHashMap<>();
    private final Map<String, RoutePlan> codeIndex = new ConcurrentHashMap<>();
    private Long idGenerator = 1L;

    @Override
    public Optional<RoutePlan> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }

    @Override
    public Optional<RoutePlan> findByPlanCode(String planCode) {
        return Optional.ofNullable(codeIndex.get(planCode));
    }

    @Override
    public List<RoutePlan> findByStatus(RoutePlan.RouteStatus status) {
        return storage.values().stream()
            .filter(plan -> plan.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public List<RoutePlan> findByVehicleCode(String vehicleCode) {
        return storage.values().stream()
            .filter(plan -> vehicleCode.equals(plan.getVehicleCode()))
            .collect(Collectors.toList());
    }

    @Override
    public List<RoutePlan> findAll() {
        return new ArrayList<>(storage.values());
    }

    @Override
    public RoutePlan save(RoutePlan routePlan) {
        if (routePlan.getId() == null) {
            routePlan.setId(idGenerator++);
        }
        storage.put(routePlan.getId(), routePlan);
        if (routePlan.getPlanCode() != null) {
            codeIndex.put(routePlan.getPlanCode(), routePlan);
        }
        return routePlan;
    }

    @Override
    public void delete(RoutePlan routePlan) {
        if (routePlan.getId() != null) {
            storage.remove(routePlan.getId());
        }
        if (routePlan.getPlanCode() != null) {
            codeIndex.remove(routePlan.getPlanCode());
        }
    }

    @Override
    public List<RoutePlan> findByPlannedStartTimeBetween(LocalDateTime start, LocalDateTime end) {
        return storage.values().stream()
            .filter(plan -> {
                LocalDateTime plannedStart = plan.getPlannedStartTime();
                return plannedStart != null && !plannedStart.isBefore(start) && !plannedStart.isAfter(end);
            })
            .collect(Collectors.toList());
    }
}