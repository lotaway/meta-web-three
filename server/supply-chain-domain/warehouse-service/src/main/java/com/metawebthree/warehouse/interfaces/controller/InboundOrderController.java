package com.metawebthree.warehouse.interfaces.controller;

import com.metawebthree.warehouse.application.WarehouseApplicationService;
import com.metawebthree.warehouse.application.dto.InboundOrderDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/inbound")
public class InboundOrderController {
    private final WarehouseApplicationService appService;

    public InboundOrderController(WarehouseApplicationService appService) {
        this.appService = appService;
    }

    @PostMapping
    public ResponseEntity<InboundOrderDTO> create(@RequestBody InboundOrderDTO dto) {
        return ResponseEntity.ok(appService.createInboundOrder(dto));
    }

    @GetMapping("/{orderNo}")
    public ResponseEntity<InboundOrderDTO> get(@PathVariable String orderNo) {
        return ResponseEntity.ok(appService.queryInboundOrder(orderNo));
    }

    @GetMapping
    public ResponseEntity<List<InboundOrderDTO>> list(
            @RequestParam(required = false) Long warehouseId,
            @RequestParam(required = false) String status) {
        return ResponseEntity.ok(appService.listInboundOrders(warehouseId, status));
    }

    @PostMapping("/{orderNo}/confirm")
    public ResponseEntity<Void> confirm(@PathVariable String orderNo) {
        appService.confirmInboundOrder(orderNo);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{orderNo}/complete")
    public ResponseEntity<Void> complete(@PathVariable String orderNo, @RequestBody InboundOrderDTO dto) {
        appService.completeInboundOrder(orderNo, dto);
        return ResponseEntity.ok().build();
    }
}