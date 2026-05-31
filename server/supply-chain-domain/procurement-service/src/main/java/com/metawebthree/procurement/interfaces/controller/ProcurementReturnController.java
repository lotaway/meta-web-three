package com.metawebthree.procurement.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.SupplyChainPermissions;
import com.metawebthree.procurement.application.ProcurementReturnService;
import com.metawebthree.procurement.application.dto.ProcurementReturnOrderDTO;
import org.springframework.web.bind.annotation.*;
import java.util.List;

/**
 * 采购退货管理 REST API
 * Capability: 采购退货单创建与审批、退货发货与物流跟踪、退货入库与库存更新、退货退款财务结算
 */
@RestController
@RequestMapping("/api/procurement/return")
public class ProcurementReturnController {

    private final ProcurementReturnService returnService;

    public ProcurementReturnController(ProcurementReturnService returnService) {
        this.returnService = returnService;
    }

    @PostMapping("/orders")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_CREATE)
    public ProcurementReturnOrderDTO createReturnOrder(@RequestBody ProcurementReturnOrderDTO dto) {
        return returnService.createReturnOrder(dto);
    }

    @PostMapping("/orders/{returnNo}/submit")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_CREATE)
    public ProcurementReturnOrderDTO submitForApproval(@PathVariable String returnNo) {
        return returnService.submitForApproval(returnNo);
    }

    @PostMapping("/orders/{returnNo}/approve")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_APPROVE)
    public ProcurementReturnOrderDTO approveReturnOrder(
            @PathVariable String returnNo,
            @RequestParam String approver,
            @RequestParam(required = false) String comment) {
        return returnService.approveReturnOrder(returnNo, approver, comment != null ? comment : "");
    }

    @PostMapping("/orders/{returnNo}/reject")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_REJECT)
    public ProcurementReturnOrderDTO rejectReturnOrder(
            @PathVariable String returnNo,
            @RequestParam String approver,
            @RequestParam String reason) {
        return returnService.rejectReturnOrder(returnNo, approver, reason);
    }

    @PostMapping("/orders/{returnNo}/ship")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_CREATE)
    public ProcurementReturnOrderDTO shipReturnOrder(
            @PathVariable String returnNo,
            @RequestParam String logisticsCompany,
            @RequestParam String trackingNumber) {
        return returnService.shipReturnOrder(returnNo, logisticsCompany, trackingNumber);
    }

    @PostMapping("/orders/{returnNo}/confirm")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_CREATE)
    public ProcurementReturnOrderDTO confirmReturned(@PathVariable String returnNo) {
        return returnService.confirmReturned(returnNo);
    }

    @PostMapping("/orders/{returnNo}/complete")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_CREATE)
    public ProcurementReturnOrderDTO completeReturnOrder(@PathVariable String returnNo) {
        return returnService.completeReturnOrder(returnNo);
    }

    @PostMapping("/orders/{returnNo}/cancel")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_CREATE)
    public ProcurementReturnOrderDTO cancelReturnOrder(@PathVariable String returnNo) {
        return returnService.cancelReturnOrder(returnNo);
    }

    @GetMapping("/orders/{returnNo}")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_READ)
    public ProcurementReturnOrderDTO queryReturnOrder(@PathVariable String returnNo) {
        return returnService.queryReturnOrder(returnNo);
    }

    @GetMapping("/orders")
    @RequirePermission(SupplyChainPermissions.PROCUREMENT_READ)
    public List<ProcurementReturnOrderDTO> listReturnOrders(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Long warehouseId,
            @RequestParam(required = false) String supplierCode) {
        return returnService.listReturnOrders(status, warehouseId, supplierCode);
    }
}