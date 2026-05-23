package com.metawebthree.routeoptimizer.domain.service;

import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import com.metawebthree.routeoptimizer.domain.entity.RoutePoint;
import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import com.metawebthree.routeoptimizer.domain.repository.RoutePlanRepository;
import com.metawebthree.routeoptimizer.domain.repository.VehicleRepository;
import java.util.Comparator;
import java.util.List;

public interface RouteOptimizerDomainService {
    RoutePlan createRoutePlan(String planName, String vehicleCode, 
        RoutePlan.OptimizationType optimizationType, List<RoutePoint> points);
    RoutePlan optimizeRoute(Long routePlanId);
    RoutePlan assignVehicleToRoute(Long routePlanId, String vehicleCode);
    RoutePlan startRouteExecution(Long routePlanId);
    RoutePlan completeRoute(Long routePlanId);
    List<Vehicle> findAvailableVehicles();
    List<RoutePlan> findRoutesByStatus(RoutePlan.RouteStatus status);
    RoutePoint addPointToRoute(Long routePlanId, RoutePoint point);
    void removePointFromRoute(Long routePlanId, Long pointId);
    RoutePlan reorderPoints(Long routePlanId, List<Integer> newSequence);
    double calculateTotalCost(RoutePlan routePlan);
    int estimateDuration(RoutePlan routePlan);
}