package com.metawebthree.routeoptimizer.infrastructure.event;

import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import com.metawebthree.routeoptimizer.domain.entity.RoutePoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

@Component
public class RouteOptimizerEventPublisher {
    private static final Logger logger = LoggerFactory.getLogger(RouteOptimizerEventPublisher.class);
    
    private final KafkaTemplate<String, String> kafkaTemplate;

    public RouteOptimizerEventPublisher(KafkaTemplate<String, String> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    public void publishRoutePlanCreated(RoutePlan plan) {
        String message = String.format("{\"event\":\"ROUTE_PLAN_CREATED\",\"planCode\":\"%s\",\"vehicleCode\":\"%s\"}",
            plan.getPlanCode(), plan.getVehicleCode());
        kafkaTemplate.send("route.optimizer.events", message);
        logger.info("Published ROUTE_PLAN_CREATED event for: {}", plan.getPlanCode());
    }

    public void publishRouteOptimized(RoutePlan plan) {
        String message = String.format("{\"event\":\"ROUTE_OPTIMIZED\",\"planCode\":\"%s\",\"totalDistance\":\"%s\"}",
            plan.getPlanCode(), plan.getTotalDistance());
        kafkaTemplate.send("route.optimizer.events", message);
        logger.info("Published ROUTE_OPTIMIZED event for: {}", plan.getPlanCode());
    }

    public void publishVehicleAssigned(RoutePlan plan) {
        String message = String.format("{\"event\":\"VEHICLE_ASSIGNED\",\"planCode\":\"%s\",\"vehicleCode\":\"%s\"}",
            plan.getPlanCode(), plan.getVehicleCode());
        kafkaTemplate.send("route.optimizer.events", message);
        logger.info("Published VEHICLE_ASSIGNED event for: {}", plan.getPlanCode());
    }

    public void publishRouteStarted(RoutePlan plan) {
        String message = String.format("{\"event\":\"ROUTE_STARTED\",\"planCode\":\"%s\"}", plan.getPlanCode());
        kafkaTemplate.send("route.optimizer.events", message);
        logger.info("Published ROUTE_STARTED event for: {}", plan.getPlanCode());
    }

    public void publishRouteCompleted(RoutePlan plan) {
        String message = String.format("{\"event\":\"ROUTE_COMPLETED\",\"planCode\":\"%s\",\"totalDistance\":\"%s\"}",
            plan.getPlanCode(), plan.getTotalDistance());
        kafkaTemplate.send("route.optimizer.events", message);
        logger.info("Published ROUTE_COMPLETED event for: {}", plan.getPlanCode());
    }

    public void publishPointAdded(Long routePlanId, RoutePoint point) {
        String message = String.format("{\"event\":\"POINT_ADDED\",\"routePlanId\":%d,\"pointCode\":\"%s\"}",
            routePlanId, point.getPointCode());
        kafkaTemplate.send("route.optimizer.events", message);
        logger.info("Published POINT_ADDED event for point: {}", point.getPointCode());
    }

    public void publishPointRemoved(Long routePlanId, Long pointId) {
        String message = String.format("{\"event\":\"POINT_REMOVED\",\"routePlanId\":%d,\"pointId\":%d}",
            routePlanId, pointId);
        kafkaTemplate.send("route.optimizer.events", message);
        logger.info("Published POINT_REMOVED event for point: {}", pointId);
    }

    public void publishRouteReordered(RoutePlan plan) {
        String message = String.format("{\"event\":\"ROUTE_REORDERED\",\"planCode\":\"%s\"}", plan.getPlanCode());
        kafkaTemplate.send("route.optimizer.events", message);
        logger.info("Published ROUTE_REORDERED event for: {}", plan.getPlanCode());
    }
}