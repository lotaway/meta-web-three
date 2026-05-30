package com.metawebthree.inventory.interfaces.controller;

import com.metawebthree.inventory.application.ReplenishmentApplicationService;
import com.metawebthree.inventory.application.dto.ReplenishmentRecommendationDTO;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/inventory/replenishment")
public class ReplenishmentController {

    private final ReplenishmentApplicationService replenishmentService;

    public ReplenishmentController(ReplenishmentApplicationService replenishmentService) {
        this.replenishmentService = replenishmentService;
    }

    @PostMapping("/generate")
    public ReplenishmentRecommendationDTO generate(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId,
            @RequestParam(required = false, defaultValue = "30") Integer daysToAnalyze) {
        return replenishmentService.generateRecommendation(skuCode, warehouseId, daysToAnalyze);
    }

    @PostMapping("/generate/warehouse/{warehouseId}")
    public List<ReplenishmentRecommendationDTO> generateForWarehouse(
            @PathVariable Long warehouseId,
            @RequestParam(required = false, defaultValue = "30") Integer daysToAnalyze) {
        return replenishmentService.generateRecommendationsForWarehouse(warehouseId, daysToAnalyze);
    }

    @GetMapping("/pending")
    public List<ReplenishmentRecommendationDTO> getPending() {
        return replenishmentService.getPendingRecommendations();
    }

    @PostMapping("/{id}/approve")
    public ReplenishmentRecommendationDTO approve(@PathVariable Long id) {
        return replenishmentService.approveRecommendation(id);
    }

    @PostMapping("/{id}/reject")
    public ReplenishmentRecommendationDTO reject(@PathVariable Long id) {
        return replenishmentService.rejectRecommendation(id);
    }

    @GetMapping("/{id}")
    public ReplenishmentRecommendationDTO queryById(@PathVariable Long id) {
        return replenishmentService.queryById(id);
    }

    @GetMapping("/warehouse/{warehouseId}")
    public List<ReplenishmentRecommendationDTO> queryByWarehouse(@PathVariable Long warehouseId) {
        return replenishmentService.queryByWarehouse(warehouseId);
    }
}