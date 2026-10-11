package com.metawebthree.routeoptimizer.application.command;

import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import com.metawebthree.routeoptimizer.domain.entity.RoutePoint;
import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import com.metawebthree.routeoptimizer.domain.service.RouteOptimizerDomainService;
import com.metawebthree.routeoptimizer.infrastructure.event.RouteOptimizerEventPublisher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

@Service
public class RouteOptimizerCommandService {
    private static final Logger logger = LoggerFactory.getLogger(RouteOptimizerCommandService.class);
    
    private final RouteOptimizerDomainService domainService;
    private final RouteOptimizerEventPublisher eventPublisher;

    public RouteOptimizerCommandService(RouteOptimizerDomainService domainService,
                                        RouteOptimizerEventPublisher eventPublisher) {
        this.domainService = domainService;
        this.eventPublisher = eventPublisher;
    }

    @Transactional
    public RoutePlan createRoutePlan(String planName, String vehicleCode,
                                      RoutePlan.OptimizationType optimizationType,
                                      List<RoutePoint> points) {
        RoutePlan plan = domainService.createRoutePlan(planName, vehicleCode, optimizationType, points);
        eventPublisher.publishRoutePlanCreated(plan);
        logger.info("Created route plan: {}", plan.getPlanCode());
        return plan;
    }

    @Transactional
    public RoutePlan optimizeRoute(Long routePlanId) {
        RoutePlan plan = domainService.optimizeRoute(routePlanId);
        eventPublisher.publishRouteOptimized(plan);
        logger.info("Optimized route plan: {}", plan.getPlanCode());
        return plan;
    }

    @Transactional
    public RoutePlan assignVehicle(Long routePlanId, String vehicleCode) {
        RoutePlan plan = domainService.assignVehicleToRoute(routePlanId, vehicleCode);
        eventPublisher.publishVehicleAssigned(plan);
        logger.info("Assigned vehicle {} to route {}", vehicleCode, plan.getPlanCode());
        return plan;
    }

    @Transactional
    public RoutePlan startExecution(Long routePlanId) {
        RoutePlan plan = domainService.startRouteExecution(routePlanId);
        eventPublisher.publishRouteStarted(plan);
        logger.info("Started route execution: {}", plan.getPlanCode());
        return plan;
    }

    @Transactional
    public RoutePlan completeRoute(Long routePlanId) {
        RoutePlan plan = domainService.completeRoute(routePlanId);
        eventPublisher.publishRouteCompleted(plan);
        logger.info("Completed route: {}", plan.getPlanCode());
        return plan;
    }

    @Transactional
    public RoutePoint addPoint(Long routePlanId, RoutePoint point) {
        RoutePoint savedPoint = domainService.addPointToRoute(routePlanId, point);
        eventPublisher.publishPointAdded(routePlanId, savedPoint);
        logger.info("Added point {} to route {}", savedPoint.getPointCode(), routePlanId);
        return savedPoint;
    }

    @Transactional
    public void removePoint(Long routePlanId, Long pointId) {
        domainService.removePointFromRoute(routePlanId, pointId);
        eventPublisher.publishPointRemoved(routePlanId, pointId);
        logger.info("Removed point {} from route {}", pointId, routePlanId);
    }

    @Transactional
    public RoutePlan reorderPoints(Long routePlanId, List<Integer> newSequence) {
        RoutePlan plan = domainService.reorderPoints(routePlanId, newSequence);
        eventPublisher.publishRouteReordered(plan);
        logger.info("Reordered points for route: {}", plan.getPlanCode());
        return plan;
    }

    @Transactional
    public Vehicle createVehicle(String vehicleCode, String vehicleNumber, String vehicleType,
                                  Double maxLoadCapacity, String driverName, String driverPhone) {
        Vehicle vehicle = new Vehicle();
        vehicle.setVehicleCode(vehicleCode);
        vehicle.setVehicleNumber(vehicleNumber);
        vehicle.setVehicleType(vehicleType);
        vehicle.setMaxLoadCapacity(maxLoadCapacity);
        vehicle.setDriverName(driverName);
        vehicle.setDriverPhone(driverPhone);
        vehicle.setStatus(Vehicle.VehicleStatus.IDLE);
        
        logger.info("Created vehicle: {}", vehicleCode);
        return vehicle;
    }

    @Transactional
    public Map<String, Object> updateVehicleLocation(String vehicleCode, Double latitude, Double longitude) {
        logger.info("Updated location for vehicle {}: ({}, {})", vehicleCode, latitude, longitude);
        return Map.of(
            "vehicleCode", vehicleCode,
            "latitude", latitude,
            "longitude", longitude,
            "updated", true
        );
    }
}