package com.metawebthree.inventoryalert.interfaces.controller;

import com.metawebthree.inventoryalert.application.dto.InventoryAlertDTO;
import com.metawebthree.inventoryalert.application.service.InventoryAlertApplicationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/inventory-alert")
public class InventoryAlertController {

    private final InventoryAlertApplicationService alertService;

    public InventoryAlertController(InventoryAlertApplicationService alertService) {
        this.alertService = alertService;
    }

    /**
     * Check inventory and create alert if needed
     */
    @PostMapping("/check")
    public ResponseEntity<Void> checkInventory(
            @RequestParam Long productId,
            @RequestParam Integer threshold) {
        alertService.checkInventory(productId, threshold);
        return ResponseEntity.ok().build();
    }

    /**
     * Resolve alert
     */
    @PostMapping("/{id}/resolve")
    public ResponseEntity<InventoryAlertDTO> resolve(
            @PathVariable Long id,
            @RequestParam(required = false) String remark) {
        InventoryAlertDTO result = alertService.processAlert(id, true, remark);
        return ResponseEntity.ok(result);
    }

    /**
     * Ignore alert
     */
    @PostMapping("/{id}/ignore")
    public ResponseEntity<InventoryAlertDTO> ignore(
            @PathVariable Long id,
            @RequestParam(required = false) String remark) {
        InventoryAlertDTO result = alertService.processAlert(id, false, remark);
        return ResponseEntity.ok(result);
    }

    /**
     * Get all alerts (admin)
     */
    @GetMapping("/list")
    public ResponseEntity<List<InventoryAlertDTO>> getAll() {
        List<InventoryAlertDTO> result = alertService.getAll();
        return ResponseEntity.ok(result);
    }

    /**
     * Get alerts by status
     */
    @GetMapping("/status/{status}")
    public ResponseEntity<List<InventoryAlertDTO>> getByStatus(@PathVariable Integer status) {
        List<InventoryAlertDTO> result = alertService.getByStatus(status);
        return ResponseEntity.ok(result);
    }

    /**
     * Get alerts by alert level
     */
    @GetMapping("/level/{alertLevel}")
    public ResponseEntity<List<InventoryAlertDTO>> getByAlertLevel(@PathVariable Integer alertLevel) {
        List<InventoryAlertDTO> result = alertService.getByAlertLevel(alertLevel);
        return ResponseEntity.ok(result);
    }

    /**
     * Get alerts by product ID
     */
    @GetMapping("/product/{productId}")
    public ResponseEntity<List<InventoryAlertDTO>> getByProductId(@PathVariable Long productId) {
        List<InventoryAlertDTO> result = alertService.getByProductId(productId);
        return ResponseEntity.ok(result);
    }

    /**
     * Get high priority alerts
     */
    @GetMapping("/high-priority")
    public ResponseEntity<List<InventoryAlertDTO>> getHighPriority() {
        List<InventoryAlertDTO> result = alertService.getHighPriorityAlerts();
        return ResponseEntity.ok(result);
    }

    /**
     * Analyze inventory turnover
     */
    @GetMapping("/turnover/{productId}")
    public ResponseEntity<InventoryAlertDTO> analyzeTurnover(@PathVariable Long productId) {
        InventoryAlertDTO result = alertService.analyzeTurnover(productId);
        return ResponseEntity.ok(result);
    }
}