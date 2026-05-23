package com.metawebthree.routeoptimizer.application.query;

import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import com.metawebthree.routeoptimizer.domain.repository.RoutePlanRepository;
import com.metawebthree.routeoptimizer.domain.repository.VehicleRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class RouteOptimizerQueryService {
    private static final Logger logger = LoggerFactory.getLogger(RouteOptimizerQueryService.class);
    
    private final RoutePlanRepository routePlanRepository;
    private final VehicleRepository vehicleRepository;

    public RouteOptimizerQueryService(RoutePlanRepository routePlanRepository,
                                      VehicleRepository vehicleRepository) {
        this.routePlanRepository = routePlanRepository;
        this.vehicleRepository = vehicleRepository;
    }

    public RoutePlan getRoutePlanById(Long id) {
        return routePlanRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Route plan not found: " + id));
    }

    public RoutePlan getRoutePlanByCode(String planCode) {
        return routePlanRepository.findByPlanCode(planCode)
            .orElseThrow(() -> new IllegalArgumentException("Route plan not found: " + planCode));
    }

    public List<RoutePlan> getAllRoutePlans() {
        return routePlanRepository.findAll();
    }

    public List<RoutePlan> getRoutePlansByStatus(RoutePlan.RouteStatus status) {
        return routePlanRepository.findByStatus(status);
    }

    public List<RoutePlan> getRoutePlansByVehicle(String vehicleCode) {
        return routePlanRepository.findByVehicleCode(vehicleCode);
    }

    public List<RoutePlan> getRoutePlansForDateRange(java.time.LocalDateTime start, 
                                                      java.time.LocalDateTime end) {
        return routePlanRepository.findByPlannedStartTimeBetween(start, end);
    }

    public Vehicle getVehicleById(Long id) {
        return vehicleRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Vehicle not found: " + id));
    }

    public Vehicle getVehicleByCode(String vehicleCode) {
        return vehicleRepository.findByVehicleCode(vehicleCode)
            .orElseThrow(() -> new IllegalArgumentException("Vehicle not found: " + vehicleCode));
    }

    public List<Vehicle> getAllVehicles() {
        return vehicleRepository.findAll();
    }

    public List<Vehicle> getVehiclesByStatus(Vehicle.VehicleStatus status) {
        return vehicleRepository.findByStatus(status);
    }

    public List<Vehicle> getAvailableVehicles() {
        return vehicleRepository.findAvailableVehicles();
    }

    public List<RoutePlan> getPendingRoutes() {
        return routePlanRepository.findByStatus(RoutePlan.RouteStatus.PENDING);
    }

    public List<RoutePlan> getInProgressRoutes() {
        return routePlanRepository.findByStatus(RoutePlan.RouteStatus.IN_PROGRESS);
    }

    public List<RoutePlan> getCompletedRoutes() {
        return routePlanRepository.findByStatus(RoutePlan.RouteStatus.COMPLETED);
    }
}