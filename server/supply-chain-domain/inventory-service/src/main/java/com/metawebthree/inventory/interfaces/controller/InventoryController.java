package com.metawebthree.inventory.interfaces.controller;

import com.metawebthree.inventory.application.InventoryApplicationService;
import com.metawebthree.inventory.application.dto.InventoryDTO;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.ReserveInventoryDTO;
import org.springframework.web.bind.annotation.*;
import java.util.List;

/**
 * 库存管理 REST API
 * 能力: 库存查询、预留、确认、取消、增减
 */
@RestController
@RequestMapping("/api/inventory")
public class InventoryController {

    private final InventoryApplicationService inventoryService;

    public InventoryController(InventoryApplicationService inventoryService) {
        this.inventoryService = inventoryService;
    }

    @GetMapping("/sku/{skuCode}")
    public List<InventoryDTO> queryBySkuCode(@PathVariable String skuCode) {
        return inventoryService.queryBySkuCode(skuCode);
    }

    @GetMapping
    public InventoryDTO queryBySku(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId) {
        return inventoryService.queryBySku(skuCode, warehouseId);
    }

    @PostMapping("/reserve")
    public InventoryOperationResult reserve(@RequestBody ReserveInventoryDTO dto) {
        return inventoryService.reserve(dto);
    }

    @PostMapping("/confirm")
    public InventoryOperationResult confirm(@RequestParam String bizId) {
        return inventoryService.confirm(bizId);
    }

    @PostMapping("/cancel")
    public InventoryOperationResult cancel(@RequestParam String bizId) {
        return inventoryService.cancel(bizId);
    }

    @PostMapping("/increase")
    public InventoryOperationResult increase(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId,
            @RequestParam Integer quantity,
            @RequestParam(required = false) String remark) {
        return inventoryService.increase(skuCode, warehouseId, quantity, remark);
    }

    @PostMapping("/decrease")
    public InventoryOperationResult decrease(
            @RequestParam String skuCode,
            @RequestParam Long warehouseId,
            @RequestParam Integer quantity,
            @RequestParam(required = false) String remark) {
        return inventoryService.decrease(skuCode, warehouseId, quantity, remark);
    }
}