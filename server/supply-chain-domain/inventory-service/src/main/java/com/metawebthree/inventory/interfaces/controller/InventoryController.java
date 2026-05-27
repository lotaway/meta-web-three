package com.metawebthree.inventory.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.inventory.application.InventoryApplicationService;
import com.metawebthree.inventory.application.dto.InventoryDTO;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.ReserveInventoryDTO;
import com.metawebthree.common.SupplyChainPermissions;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/inventory")
public class InventoryController {

    private final InventoryApplicationService inventoryService;

    public InventoryController(InventoryApplicationService inventoryService) {
        this.inventoryService = inventoryService;
    }

    @RequirePermission(SupplyChainPermissions.INVENTORY_READ)
    @GetMapping("/sku/{skuCode}")
    public List<InventoryDTO> queryBySkuCode(
            @PathVariable String skuCode,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return inventoryService.queryBySkuCode(skuCode);
    }

    @RequirePermission(SupplyChainPermissions.INVENTORY_READ)
    @GetMapping
    public InventoryDTO queryBySku(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return inventoryService.queryBySku(skuCode, warehouseId);
    }

    @RequirePermission(SupplyChainPermissions.INVENTORY_RESERVE)
    @PostMapping("/reserve")
    public InventoryOperationResult reserve(
            @RequestBody ReserveInventoryDTO dto,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return inventoryService.reserve(dto);
    }

    @RequirePermission(SupplyChainPermissions.INVENTORY_CONFIRM)
    @PostMapping("/confirm")
    public InventoryOperationResult confirm(
            @RequestParam String bizId,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return inventoryService.confirm(bizId);
    }

    @RequirePermission(SupplyChainPermissions.INVENTORY_CANCEL)
    @PostMapping("/cancel")
    public InventoryOperationResult cancel(
            @RequestParam String bizId,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return inventoryService.cancel(bizId);
    }

    @RequirePermission(SupplyChainPermissions.INVENTORY_ADJUST)
    @PostMapping("/increase")
    public InventoryOperationResult increase(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId,
            @RequestParam Integer quantity,
            @RequestParam(required = false) String remark,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return inventoryService.increase(skuCode, warehouseId, quantity, remark);
    }

    @RequirePermission(SupplyChainPermissions.INVENTORY_ADJUST)
    @PostMapping("/decrease")
    public InventoryOperationResult decrease(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId,
            @RequestParam Integer quantity,
            @RequestParam(required = false) String remark,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return inventoryService.decrease(skuCode, warehouseId, quantity, remark);
    }
}