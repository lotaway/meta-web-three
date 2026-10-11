package com.metawebthree.routeoptimizer.interfaces.controller;

import com.metawebthree.routeoptimizer.application.command.RouteOptimizerCommandService;
import com.metawebthree.routeoptimizer.application.query.RouteOptimizerQueryService;
import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import com.metawebthree.routeoptimizer.domain.entity.RoutePoint;
import com.metawebthree.routeoptimizer.domain.entity.Vehicle;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/route-optimizer")
public class RouteOptimizerController {
    
    private final RouteOptimizerCommandService commandService;
    private final RouteOptimizerQueryService queryService;

    public RouteOptimizerController(RouteOptimizerCommandService commandService,
                                    RouteOptimizerQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping("/routes")
    public ResponseEntity<RoutePlan> createRoutePlan(@RequestBody Map<String, Object> request) {
        String planName = (String) request.get("planName");
        String vehicleCode = (String) request.get("vehicleCode");
        String optTypeStr = (String) request.get("optimizationType");
        RoutePlan.OptimizationType optType = RoutePlan.OptimizationType.valueOf(optTypeStr);
        
        RoutePlan plan = commandService.createRoutePlan(planName, vehicleCode, optType, List.of());
        return ResponseEntity.ok(plan);
    }

    @GetMapping("/routes/{id}")
    public ResponseEntity<RoutePlan> getRoutePlan(@PathVariable Long id) {
        return ResponseEntity.ok(queryService.getRoutePlanById(id));
    }

    @GetMapping("/routes")
    public ResponseEntity<List<RoutePlan>> getAllRoutePlans() {
        return ResponseEntity.ok(queryService.getAllRoutePlans());
    }

    @GetMapping("/routes/status/{status}")
    public ResponseEntity<List<RoutePlan>> getRoutesByStatus(@PathVariable RoutePlan.RouteStatus status) {
        return ResponseEntity.ok(queryService.getRoutePlansByStatus(status));
    }

    @PostMapping("/routes/{id}/optimize")
    public ResponseEntity<RoutePlan> optimizeRoute(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.optimizeRoute(id));
    }

    @PostMapping("/routes/{id}/assign")
    public ResponseEntity<RoutePlan> assignVehicle(@PathVariable Long id, 
                                                    @RequestBody Map<String, String> request) {
        String vehicleCode = request.get("vehicleCode");
        return ResponseEntity.ok(commandService.assignVehicle(id, vehicleCode));
    }

    @PostMapping("/routes/{id}/start")
    public ResponseEntity<RoutePlan> startRoute(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.startExecution(id));
    }

    @PostMapping("/routes/{id}/complete")
    public ResponseEntity<RoutePlan> completeRoute(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.completeRoute(id));
    }

    @PostMapping("/routes/{id}/points")
    public ResponseEntity<RoutePoint> addPoint(@PathVariable Long id, @RequestBody RoutePoint point) {
        return ResponseEntity.ok(commandService.addPoint(id, point));
    }

    @DeleteMapping("/routes/{routeId}/points/{pointId}")
    public ResponseEntity<Void> removePoint(@PathVariable Long routeId, @PathVariable Long pointId) {
        commandService.removePoint(routeId, pointId);
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/vehicles")
    public ResponseEntity<Vehicle> createVehicle(@RequestBody Map<String, Object> request) {
        Vehicle vehicle = commandService.createVehicle(
            (String) request.get("vehicleCode"),
            (String) request.get("vehicleNumber"),
            (String) request.get("vehicleType"),
            ((Number) request.get("maxLoadCapacity")).doubleValue(),
            (String) request.get("driverName"),
            (String) request.get("driverPhone")
        );
        return ResponseEntity.ok(vehicle);
    }

    @GetMapping("/vehicles/{id}")
    public ResponseEntity<Vehicle> getVehicle(@PathVariable Long id) {
        return ResponseEntity.ok(queryService.getVehicleById(id));
    }

    @GetMapping("/vehicles")
    public ResponseEntity<List<Vehicle>> getAllVehicles() {
        return ResponseEntity.ok(queryService.getAllVehicles());
    }

    @GetMapping("/vehicles/available")
    public ResponseEntity<List<Vehicle>> getAvailableVehicles() {
        return ResponseEntity.ok(queryService.getAvailableVehicles());
    }

    @GetMapping("/vehicles/status/{status}")
    public ResponseEntity<List<Vehicle>> getVehiclesByStatus(@PathVariable Vehicle.VehicleStatus status) {
        return ResponseEntity.ok(queryService.getVehiclesByStatus(status));
    }

    @PutMapping("/vehicles/{code}/location")
    public ResponseEntity<Map<String, Object>> updateVehicleLocation(
            @PathVariable String code, @RequestBody Map<String, Double> request) {
        Double latitude = request.get("latitude");
        Double longitude = request.get("longitude");
        return ResponseEntity.ok(commandService.updateVehicleLocation(code, latitude, longitude));
    }

    @GetMapping("/dashboard/pending-routes")
    public ResponseEntity<List<RoutePlan>> getPendingRoutes() {
        return ResponseEntity.ok(queryService.getPendingRoutes());
    }

    @GetMapping("/dashboard/in-progress-routes")
    public ResponseEntity<List<RoutePlan>> getInProgressRoutes() {
        return ResponseEntity.ok(queryService.getInProgressRoutes());
    }

    @GetMapping("/dashboard/completed-routes")
    public ResponseEntity<List<RoutePlan>> getCompletedRoutes() {
        return ResponseEntity.ok(queryService.getCompletedRoutes());
    }
}