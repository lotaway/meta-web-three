package com.metawebthree.warehouse.interfaces.controller;

import com.metawebthree.warehouse.application.WarehouseApplicationService;
import com.metawebthree.warehouse.application.dto.WarehouseDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderDTO;
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
    public WarehouseDTO createWarehouse(@RequestBody WarehouseDTO dto) {
        return warehouseService.createWarehouse(dto);
    }

    @PutMapping("/warehouses/{id}")
    public WarehouseDTO updateWarehouse(
            @PathVariable Long id,
            @RequestBody WarehouseDTO dto) {
        return warehouseService.updateWarehouse(id, dto);
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
    public InboundOrderDTO createInboundOrder(@RequestBody InboundOrderDTO dto) {
        return warehouseService.createInboundOrder(dto);
    }

    @PostMapping("/inbound/{orderNo}/confirm")
    public InboundOrderDTO confirmInboundOrder(@PathVariable String orderNo) {
        return warehouseService.confirmInboundOrder(orderNo);
    }

    @PostMapping("/inbound/{orderNo}/complete")
    public InboundOrderDTO completeInboundOrder(
            @PathVariable String orderNo,
            @RequestBody InboundOrderDTO dto) {
        return warehouseService.completeInboundOrder(orderNo, dto);
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