package com.metawebthree.warehouse.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.warehouse.application.WarehouseApplicationService;
import com.metawebthree.warehouse.application.dto.WarehouseDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderDTO;
import com.metawebthree.common.SupplyChainPermissions;
import com.metawebthree.common.dto.ApiResponse;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/warehouse")
public class WarehouseController {

    private final WarehouseApplicationService warehouseService;

    public WarehouseController(WarehouseApplicationService warehouseService) {
        this.warehouseService = warehouseService;
    }

    // 仓库管理
    @RequirePermission(SupplyChainPermissions.WAREHOUSE_CREATE)
    @PostMapping("/warehouses")
    public ApiResponse<Long> createWarehouse(
            @RequestBody WarehouseDTO dto,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        Long id = warehouseService.createWarehouse(dto);
        return ApiResponse.success(id);
    }

    @RequirePermission(SupplyChainPermissions.WAREHOUSE_UPDATE)
    @PutMapping("/warehouses/{id}")
    public ApiResponse<Void> updateWarehouse(
            @PathVariable Long id,
            @RequestBody WarehouseDTO dto,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        warehouseService.updateWarehouse(id, dto);
        return ApiResponse.success();
    }

    @RequirePermission(SupplyChainPermissions.WAREHOUSE_READ)
    @GetMapping("/warehouses/{id}")
    public WarehouseDTO queryWarehouse(
            @PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return warehouseService.queryWarehouse(id);
    }

    @RequirePermission(SupplyChainPermissions.WAREHOUSE_READ)
    @GetMapping("/warehouses")
    public List<WarehouseDTO> listWarehouses(
            @RequestParam(required = false) String status,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return warehouseService.listWarehouses(status);
    }

    // 入库管理
    @RequirePermission(SupplyChainPermissions.INBOUND_CREATE)
    @PostMapping("/inbound")
    public ApiResponse<String> createInboundOrder(
            @RequestBody InboundOrderDTO dto,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        InboundOrderDTO result = warehouseService.createInboundOrder(dto);
        return ApiResponse.success(result != null ? result.getOrderNo() : null);
    }

    @RequirePermission(SupplyChainPermissions.INBOUND_CONFIRM)
    @PostMapping("/inbound/{orderNo}/confirm")
    public ApiResponse<Void> confirmInboundOrder(
            @PathVariable String orderNo,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        warehouseService.confirmInboundOrder(orderNo);
        return ApiResponse.success();
    }

    @RequirePermission(SupplyChainPermissions.INBOUND_COMPLETE)
    @PostMapping("/inbound/{orderNo}/complete")
    public ApiResponse<Void> completeInboundOrder(
            @PathVariable String orderNo,
            @RequestBody InboundOrderDTO dto,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        warehouseService.completeInboundOrder(orderNo, dto);
        return ApiResponse.success();
    }

    @RequirePermission(SupplyChainPermissions.INBOUND_READ)
    @GetMapping("/inbound/{orderNo}")
    public InboundOrderDTO queryInboundOrder(
            @PathVariable String orderNo,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return warehouseService.queryInboundOrder(orderNo);
    }

    @RequirePermission(SupplyChainPermissions.INBOUND_READ)
    @GetMapping("/inbound")
    public List<InboundOrderDTO> listInboundOrders(
            @RequestParam(required = false) Long warehouseId,
            @RequestParam(required = false) String status,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return warehouseService.listInboundOrders(warehouseId, status);
    }
}