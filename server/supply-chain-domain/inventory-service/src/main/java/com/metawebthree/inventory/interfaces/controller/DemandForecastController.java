package com.metawebthree.inventory.interfaces.controller;

import com.metawebthree.inventory.application.DemandForecastApplicationService;
import com.metawebthree.inventory.application.dto.DemandForecastDTO;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/inventory/forecast")
public class DemandForecastController {

    private final DemandForecastApplicationService forecastService;

    public DemandForecastController(DemandForecastApplicationService forecastService) {
        this.forecastService = forecastService;
    }

    @PostMapping("/generate")
    public DemandForecastDTO generate(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId,
            @RequestParam(required = false, defaultValue = "30") Integer forecastDays,
            @RequestParam(required = false, defaultValue = "SMA") String method) {
        return forecastService.generateForecast(skuCode, warehouseId, forecastDays, method);
    }

    @PostMapping("/generate/warehouse/{warehouseId}")
    public List<DemandForecastDTO> generateForWarehouse(
            @PathVariable Long warehouseId,
            @RequestParam(required = false, defaultValue = "30") Integer forecastDays,
            @RequestParam(required = false, defaultValue = "SMA") String method) {
        return forecastService.generateForecastsForWarehouse(warehouseId, forecastDays, method);
    }

    @GetMapping("/pending")
    public List<DemandForecastDTO> getPending() {
        return forecastService.getPendingForecasts();
    }

    @PostMapping("/{id}/approve")
    public DemandForecastDTO approve(@PathVariable Long id) {
        return forecastService.approveForecast(id);
    }

    @PostMapping("/{id}/reject")
    public DemandForecastDTO reject(@PathVariable Long id) {
        return forecastService.rejectForecast(id);
    }

    @GetMapping("/{id}")
    public DemandForecastDTO queryById(@PathVariable Long id) {
        return forecastService.queryById(id);
    }

    @GetMapping("/warehouse/{warehouseId}")
    public List<DemandForecastDTO> queryByWarehouse(@PathVariable Long warehouseId) {
        return forecastService.queryByWarehouse(warehouseId);
    }
}