package com.metawebthree.production.interfaces.controller;

import com.metawebthree.production.application.command.ProductionCommandService;
import com.metawebthree.production.application.query.ProductionQueryService;
import com.metawebthree.production.domain.entity.ProductionOrder;
import com.metawebthree.production.domain.entity.WorkStation;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/production")
public class ProductionController {
    
    private final ProductionCommandService commandService;
    private final ProductionQueryService queryService;

    public ProductionController(ProductionCommandService commandService,
                                 ProductionQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    // Production Order APIs
    @PostMapping("/orders")
    public ResponseEntity<ProductionOrder> createOrder(@RequestBody Map<String, Object> request) {
        String productCode = (String) request.get("productCode");
        String productName = (String) request.get("productName");
        Integer quantity = ((Number) request.get("quantity")).intValue();
        String priorityStr = (String) request.get("priority");
        ProductionOrder.Priority priority = priorityStr != null 
            ? ProductionOrder.Priority.valueOf(priorityStr) 
            : ProductionOrder.Priority.NORMAL;
        String workshopCode = (String) request.get("workshopCode");
        
        return ResponseEntity.ok(commandService.createOrder(
            productCode, productName, quantity, priority, workshopCode));
    }

    @GetMapping("/orders/{id}")
    public ResponseEntity<ProductionOrder> getOrder(@PathVariable Long id) {
        return ResponseEntity.ok(queryService.getOrderById(id));
    }

    @GetMapping("/orders")
    public ResponseEntity<List<ProductionOrder>> getAllOrders() {
        return ResponseEntity.ok(queryService.getAllOrders());
    }

    @GetMapping("/orders/status/{status}")
    public ResponseEntity<List<ProductionOrder>> getOrdersByStatus(@PathVariable ProductionOrder.OrderStatus status) {
        return ResponseEntity.ok(queryService.getOrdersByStatus(status));
    }

    @PostMapping("/orders/{id}/schedule")
    public ResponseEntity<ProductionOrder> scheduleOrder(@PathVariable Long id, 
                                                          @RequestBody Map<String, String> request) {
        String productionLineCode = request.get("productionLineCode");
        return ResponseEntity.ok(commandService.scheduleOrder(id, productionLineCode));
    }

    @PostMapping("/orders/{id}/start")
    public ResponseEntity<ProductionOrder> startProduction(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.startProduction(id));
    }

    @PostMapping("/orders/{id}/pause")
    public ResponseEntity<ProductionOrder> pauseProduction(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.pauseProduction(id));
    }

    @PostMapping("/orders/{id}/resume")
    public ResponseEntity<ProductionOrder> resumeProduction(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.resumeProduction(id));
    }

    @PostMapping("/orders/{id}/complete")
    public ResponseEntity<ProductionOrder> completeProduction(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.completeProduction(id));
    }

    @PostMapping("/orders/{id}/cancel")
    public ResponseEntity<ProductionOrder> cancelOrder(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.cancelOrder(id));
    }

    // Work Station APIs
    @PostMapping("/stations")
    public ResponseEntity<WorkStation> createWorkStation(@RequestBody Map<String, Object> request) {
        return ResponseEntity.ok(commandService.createWorkStation(
            (String) request.get("stationCode"),
            (String) request.get("stationName"),
            (String) request.get("stationType"),
            (String) request.get("workshopCode"),
            ((Number) request.get("capacity")).intValue()
        ));
    }

    @GetMapping("/stations/{id}")
    public ResponseEntity<WorkStation> getStation(@PathVariable Long id) {
        return ResponseEntity.ok(queryService.getStationById(id));
    }

    @GetMapping("/stations")
    public ResponseEntity<List<WorkStation>> getAllStations() {
        return ResponseEntity.ok(queryService.getAllStations());
    }

    @GetMapping("/stations/available")
    public ResponseEntity<List<WorkStation>> getAvailableStations() {
        return ResponseEntity.ok(queryService.getAvailableStations());
    }

    @PostMapping("/stations/{code}/assign")
    public ResponseEntity<WorkStation> assignOrder(@PathVariable String code, 
                                                    @RequestBody Map<String, String> request) {
        String orderCode = request.get("orderCode");
        return ResponseEntity.ok(commandService.assignOrderToStation(code, orderCode));
    }
}