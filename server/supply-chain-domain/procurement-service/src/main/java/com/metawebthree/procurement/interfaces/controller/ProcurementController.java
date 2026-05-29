package com.metawebthree.procurement.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.SupplyChainPermissions;
import com.metawebthree.procurement.application.ProcurementApplicationService;
import com.metawebthree.procurement.application.dto.ProcurementOrderDTO;
import org.springframework.web.bind.annotation.*;
import java.util.List;

/**
 * 采购管理 REST API
 * 能力: 采购订单、供应商管理(关联)、采购入库
 */
@RestController
@RequestMapping("/api/procurement")
public class ProcurementController {

    private final ProcurementApplicationService procurementService;

    public ProcurementController(ProcurementApplicationService procurementService) {
        this.procurementService = procurementService;
    }

    @PostMapping("/orders")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_CREATE)
    public ProcurementOrderDTO createOrder(@RequestBody ProcurementOrderDTO dto) {
        return procurementService.createOrder(dto);
    }

    @GetMapping("/orders/{orderNo}")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_READ)
    public ProcurementOrderDTO queryOrder(@PathVariable String orderNo) {
        return procurementService.queryOrder(orderNo);
    }

    @PostMapping("/orders/{orderNo}/approve")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_APPROVE)
    public ProcurementOrderDTO approveOrder(
            @PathVariable String orderNo,
            @RequestParam String approver) {
        return procurementService.approveOrder(orderNo, approver);
    }

    @PostMapping("/orders/{orderNo}/reject")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_REJECT)
    public ProcurementOrderDTO rejectOrder(
            @PathVariable String orderNo,
            @RequestParam String reason) {
        return procurementService.rejectOrder(orderNo, reason);
    }

    @GetMapping("/orders")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_READ)
    public List<ProcurementOrderDTO> listOrders(@RequestParam(required = false) String status) {
        return procurementService.listOrders(status);
    }
}