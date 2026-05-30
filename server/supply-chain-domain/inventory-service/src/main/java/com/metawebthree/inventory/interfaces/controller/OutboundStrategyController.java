package com.metawebthree.inventory.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.inventory.application.OutboundStrategyApplicationService;
import com.metawebthree.inventory.application.dto.BatchAllocationDTO;
import com.metawebthree.inventory.application.dto.OutboundStrategyDTO;
import com.metawebthree.common.SupplyChainPermissions;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/outbound-strategy")
public class OutboundStrategyController {

    private final OutboundStrategyApplicationService service;

    public OutboundStrategyController(OutboundStrategyApplicationService service) {
        this.service = service;
    }

    @RequirePermission(SupplyChainPermissions.OUTBOUND_STRATEGY_READ)
    @GetMapping("/{id}")
    public OutboundStrategyDTO getById(@PathVariable Long id) {
        return service.getStrategyById(id);
    }

    @RequirePermission(SupplyChainPermissions.OUTBOUND_STRATEGY_READ)
    @GetMapping("/list")
    public List<OutboundStrategyDTO> list(
            @RequestParam(required = false) Long warehouseId,
            @RequestParam(required = false) Boolean isActive) {
        return service.listStrategies(warehouseId, isActive);
    }

    @RequirePermission(SupplyChainPermissions.OUTBOUND_STRATEGY_READ)
    @GetMapping("/effective")
    public OutboundStrategyDTO getEffectiveStrategy(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId) {
        return service.getEffectiveStrategy(skuCode, warehouseId);
    }

    @RequirePermission(SupplyChainPermissions.OUTBOUND_STRATEGY_CREATE)
    @PostMapping
    public OutboundStrategyDTO create(
            @RequestBody OutboundStrategyDTO dto,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId) {
        dto.setCreator(userId);
        return service.createStrategy(dto);
    }

    @RequirePermission(SupplyChainPermissions.OUTBOUND_STRATEGY_UPDATE)
    @PutMapping("/{id}")
    public OutboundStrategyDTO update(
            @PathVariable Long id,
            @RequestBody OutboundStrategyDTO dto) {
        dto.setId(id);
        return service.updateStrategy(dto);
    }

    @RequirePermission(SupplyChainPermissions.OUTBOUND_STRATEGY_DELETE)
    @DeleteMapping("/{id}")
    public boolean delete(@PathVariable Long id) {
        return service.deleteStrategy(id);
    }

    @RequirePermission(SupplyChainPermissions.OUTBOUND_ALLOCATE)
    @PostMapping("/allocate")
    public BatchAllocationDTO allocate(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId,
            @RequestParam Integer quantity,
            @RequestParam(required = false, defaultValue = "FIFO") String strategyType) {
        return service.allocateBatches(skuCode, warehouseId, quantity, strategyType);
    }

    @RequirePermission(SupplyChainPermissions.OUTBOUND_ALLOCATE)
    @PostMapping("/allocate/strategy/{strategyId}")
    public BatchAllocationDTO allocateByStrategy(
            @PathVariable Long strategyId,
            @RequestParam String skuCode,
            @RequestParam Long warehouseId,
            @RequestParam Integer quantity) {
        return service.allocateBatchesByStrategyId(strategyId, skuCode, warehouseId, quantity);
    }
}