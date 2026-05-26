package com.metawebthree.warehouse.interfaces.controller;

import com.metawebthree.warehouse.application.WarehouseApplicationService;
import com.metawebthree.warehouse.application.dto.WarehouseDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderDTO;
import com.metawebthree.common.dto.ApiResponse;
import org.springframework.web.bind.annotation.*;
import java.util.List;

/**
 * 仓库管理 REST API
 * 能力: 仓库管理、库位管理、出入库管理
 */
@RestController
@RequestMapping("/api/warehouse")
public class WarehouseController {

    private final WarehouseApplicationService warehouseService;

    public WarehouseController(WarehouseApplicationService warehouseService) {
        this.warehouseService = warehouseService;
    }

    // 仓库管理
    @PostMapping("/warehouses")
    public ApiResponse<Long> createWarehouse(@RequestBody WarehouseDTO dto) {
        Long id = warehouseService.createWarehouse(dto);
        return ApiResponse.success(id);
    }

    @PutMapping("/warehouses/{id}")
    public ApiResponse<Void> updateWarehouse(
            @PathVariable Long id,
            @RequestBody WarehouseDTO dto) {
        warehouseService.updateWarehouse(id, dto);
        return ApiResponse.success();
    }

    @GetMapping("/warehouses/{id}")
    public WarehouseDTO queryWarehouse(@PathVariable Long id) {
        return warehouseService.queryWarehouse(id);
    }

    @GetMapping("/warehouses")
    public List<WarehouseDTO> listWarehouses(@RequestParam(required = false) String status) {
        return warehouseService.listWarehouses(status);
    }

    // 入库管理
    @PostMapping("/inbound")
    public ApiResponse<Void> createInboundOrder(@RequestBody InboundOrderDTO dto) {
        warehouseService.createInboundOrder(dto);
        return ApiResponse.success();
    }

    @PostMapping("/inbound/{orderNo}/confirm")
    public ApiResponse<Void> confirmInboundOrder(@PathVariable String orderNo) {
        warehouseService.confirmInboundOrder(orderNo);
        return ApiResponse.success();
    }

    @PostMapping("/inbound/{orderNo}/complete")
    public ApiResponse<Void> completeInboundOrder(
            @PathVariable String orderNo,
            @RequestBody InboundOrderDTO dto) {
        warehouseService.completeInboundOrder(orderNo, dto);
        return ApiResponse.success();
    }

    @GetMapping("/inbound/{orderNo}")
    public InboundOrderDTO queryInboundOrder(@PathVariable String orderNo) {
        return warehouseService.queryInboundOrder(orderNo);
    }

    @GetMapping("/inbound")
    public List<InboundOrderDTO> listInboundOrders(
            @RequestParam(required = false) Long warehouseId,
            @RequestParam(required = false) String status) {
        return warehouseService.listInboundOrders(warehouseId, status);
    }
}