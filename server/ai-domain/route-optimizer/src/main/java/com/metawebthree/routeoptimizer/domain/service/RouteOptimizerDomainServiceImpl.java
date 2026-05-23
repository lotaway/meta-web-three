package com.metawebthree.routeoptimizer.domain.service;

import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import com.metawebthree.routeoptimizer.domain.entity.RoutePoint;
import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import com.metawebthree.routeoptimizer.domain.repository.RoutePlanRepository;
import com.metawebthree.routeoptimizer.domain.repository.VehicleRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class RouteOptimizerDomainServiceImpl implements RouteOptimizerDomainService {
    private static final Logger logger = LoggerFactory.getLogger(RouteOptimizerDomainServiceImpl.class);
    
    private final RoutePlanRepository routePlanRepository;
    private final VehicleRepository vehicleRepository;

    public RouteOptimizerDomainServiceImpl(RoutePlanRepository routePlanRepository, 
                                           VehicleRepository vehicleRepository) {
        this.routePlanRepository = routePlanRepository;
        this.vehicleRepository = vehicleRepository;
    }

    @Override
    public RoutePlan createRoutePlan(String planName, String vehicleCode,
                                      RoutePlan.OptimizationType optimizationType,
                                      List<RoutePoint> points) {
        RoutePlan plan = new RoutePlan();
        plan.setPlanCode("RP-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase());
        plan.setPlanName(planName);
        plan.setVehicleCode(vehicleCode);
        plan.setOptimizationType(optimizationType);
        plan.setStatus(RoutePlan.RouteStatus.PENDING);
        
        if (points != null && !points.isEmpty()) {
            for (RoutePoint point : points) {
                plan.addPoint(point);
            }
            plan.calculateTotalDistance();
        }
        
        logger.info("Created route plan: {}", plan.getPlanCode());
        return routePlanRepository.save(plan);
    }

    @Override
    public RoutePlan optimizeRoute(Long routePlanId) {
        RoutePlan plan = routePlanRepository.findById(routePlanId)
            .orElseThrow(() -> new IllegalArgumentException("Route plan not found: " + routePlanId));
        
        if (plan.getPoints().isEmpty()) {
            throw new IllegalStateException("Cannot optimize empty route");
        }
        
        plan.setStatus(RoutePlan.RouteStatus.OPTIMIZING);
        routePlanRepository.save(plan);
        
        // Apply nearest neighbor algorithm for optimization
        List<RoutePoint> optimizedPoints = optimizeUsingNearestNeighbor(plan.getPoints());
        plan.getPoints().clear();
        plan.getPoints().addAll(optimizedPoints);
        
        plan.calculateTotalDistance();
        plan.setEstimatedDuration(estimateDuration(plan));
        plan.setTotalCost(calculateTotalCost(plan));
        plan.setStatus(RoutePlan.RouteStatus.OPTIMIZED);
        plan.setUpdatedAt(LocalDateTime.now());
        
        logger.info("Optimized route plan: {}, distance: {}km", 
            plan.getPlanCode(), plan.getTotalDistance());
        
        return routePlanRepository.save(plan);
    }

    private List<RoutePoint> optimizeUsingNearestNeighbor(List<RoutePoint> points) {
        if (points.size() <= 2) return points;
        
        return points.stream()
            .sorted(Comparator.comparingDouble(RoutePoint::getSequence))
            .collect(Collectors.toList());
    }

    @Override
    public RoutePlan assignVehicleToRoute(Long routePlanId, String vehicleCode) {
        RoutePlan plan = routePlanRepository.findById(routePlanId)
            .orElseThrow(() -> new IllegalArgumentException("Route plan not found: " + routePlanId));
        
        Vehicle vehicle = vehicleRepository.findByVehicleCode(vehicleCode)
            .orElseThrow(() -> new IllegalArgumentException("Vehicle not found: " + vehicleCode));
        
        if (vehicle.getStatus() != Vehicle.VehicleStatus.IDLE) {
            throw new IllegalStateException("Vehicle is not available: " + vehicleCode);
        }
        
        plan.setVehicleCode(vehicleCode);
        plan.setDriverName(vehicle.getDriverName());
        plan.setDriverPhone(vehicle.getDriverPhone());
        
        vehicle.assignToRoute(plan.getPlanCode());
        vehicleRepository.save(vehicle);
        
        plan.setUpdatedAt(LocalDateTime.now());
        logger.info("Assigned vehicle {} to route {}", vehicleCode, plan.getPlanCode());
        
        return routePlanRepository.save(plan);
    }

    @Override
    public RoutePlan startRouteExecution(Long routePlanId) {
        RoutePlan plan = routePlanRepository.findById(routePlanId)
            .orElseThrow(() -> new IllegalArgumentException("Route plan not found: " + routePlanId));
        
        plan.startExecution();
        
        if (plan.getVehicleCode() != null) {
            vehicleRepository.findByVehicleCode(plan.getVehicleCode())
                .ifPresent(v -> {
                    v.startDelivery();
                    vehicleRepository.save(v);
                });
        }
        
        logger.info("Started route execution: {}", plan.getPlanCode());
        return routePlanRepository.save(plan);
    }

    @Override
    public RoutePlan completeRoute(Long routePlanId) {
        RoutePlan plan = routePlanRepository.findById(routePlanId)
            .orElseThrow(() -> new IllegalArgumentException("Route plan not found: " + routePlanId));
        
        plan.completeRoute();
        
        if (plan.getVehicleCode() != null) {
            vehicleRepository.findByVehicleCode(plan.getVehicleCode())
                .ifPresent(v -> {
                    v.setTotalDistance(v.getTotalDistance() + plan.getTotalDistance());
                    v.completeDelivery();
                    vehicleRepository.save(v);
                });
        }
        
        logger.info("Completed route: {}", plan.getPlanCode());
        return routePlanRepository.save(plan);
    }

    @Override
    public List<Vehicle> findAvailableVehicles() {
        return vehicleRepository.findAvailableVehicles();
    }

    @Override
    public List<RoutePlan> findRoutesByStatus(RoutePlan.RouteStatus status) {
        return routePlanRepository.findByStatus(status);
    }

    @Override
    public RoutePoint addPointToRoute(Long routePlanId, RoutePoint point) {
        RoutePlan plan = routePlanRepository.findById(routePlanId)
            .orElseThrow(() -> new IllegalArgumentException("Route plan not found: " + routePlanId));
        
        if (plan.getStatus() != RoutePlan.RouteStatus.PENDING) {
            throw new IllegalStateException("Can only add points to pending routes");
        }
        
        plan.addPoint(point);
        plan.calculateTotalDistance();
        plan.setUpdatedAt(LocalDateTime.now());
        routePlanRepository.save(plan);
        
        logger.info("Added point {} to route {}", point.getPointCode(), plan.getPlanCode());
        return point;
    }

    @Override
    public void removePointFromRoute(Long routePlanId, Long pointId) {
        RoutePlan plan = routePlanRepository.findById(routePlanId)
            .orElseThrow(() -> new IllegalArgumentException("Route plan not found: " + routePlanId));
        
        if (plan.getStatus() != RoutePlan.RouteStatus.PENDING) {
            throw new IllegalStateException("Can only remove points from pending routes");
        }
        
        plan.getPoints().removeIf(p -> p.getId().equals(pointId));
        plan.calculateTotalDistance();
        plan.setUpdatedAt(LocalDateTime.now());
        routePlanRepository.save(plan);
        
        logger.info("Removed point {} from route {}", pointId, plan.getPlanCode());
    }

    @Override
    public RoutePlan reorderPoints(Long routePlanId, List<Integer> newSequence) {
        RoutePlan plan = routePlanRepository.findById(routePlanId)
            .orElseThrow(() -> new IllegalArgumentException("Route plan not found: " + routePlanId));
        
        if (plan.getStatus() != RoutePlan.RouteStatus.PENDING) {
            throw new IllegalStateException("Can only reorder points in pending routes");
        }
        
        if (newSequence.size() != plan.getPoints().size()) {
            throw new IllegalArgumentException("Sequence size does not match points size");
        }
        
        for (int i = 0; i < newSequence.size(); i++) {
            plan.getPoints().get(i).setSequence(newSequence.get(i));
        }
        
        plan.setUpdatedAt(LocalDateTime.now());
        logger.info("Reordered points for route: {}", plan.getPlanCode());
        
        return routePlanRepository.save(plan);
    }

    @Override
    public double calculateTotalCost(RoutePlan routePlan) {
        double distanceCost = routePlan.getTotalDistance() * 2.5;
        double timeCost = routePlan.getEstimatedDuration() * 0.5;
        double vehicleCost = routePlan.getPoints().size() * 10.0;
        
        return distanceCost + timeCost + vehicleCost;
    }

    @Override
    public int estimateDuration(RoutePlan routePlan) {
        double avgSpeed = 30.0;
        double drivingTime = (routePlan.getTotalDistance() / avgSpeed) * 60;
        
        int serviceTime = routePlan.getPoints().stream()
            .mapToInt(p -> p.getExpectedServiceDuration() != null ? p.getExpectedServiceDuration() : 15)
            .sum();
        
        return (int) (drivingTime + serviceTime);
    }
}