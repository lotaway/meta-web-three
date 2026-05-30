package com.metawebthree.supplier.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.SupplyChainPermissions;
import com.metawebthree.supplier.application.SupplierPortalApplicationService;
import com.metawebthree.supplier.application.dto.SupplierPortalOrderDTO;
import com.metawebthree.supplier.application.dto.SupplierReconciliationDTO;
import com.metawebthree.supplier.application.dto.SupplierShipmentNoticeDTO;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 供应商协同门户 REST API
 * 能力: 供应商自助查询订单、发货通知、对账
 */
@RestController
@RequestMapping("/api/supplier-portal")
public class SupplierPortalController {

    private final SupplierPortalApplicationService portalService;

    public SupplierPortalController(SupplierPortalApplicationService portalService) {
        this.portalService = portalService;
    }

    // ==================== 订单查询 ====================
    @GetMapping("/orders")
    @RequirePermission(SupplyChainPermissions.PORTAL_ORDER_READ)
    public List<SupplierPortalOrderDTO> queryOrders(
            @RequestParam String supplierCode,
            @RequestParam(required = false) String status) {
        return portalService.queryOrdersBySupplier(supplierCode, status);
    }

    @GetMapping("/orders/{orderNo}")
    @RequirePermission(SupplyChainPermissions.PORTAL_ORDER_READ)
    public SupplierPortalOrderDTO queryOrderDetail(@PathVariable String orderNo) {
        return portalService.queryOrderDetail(orderNo);
    }

    // ==================== 发货通知 ====================
    @PostMapping("/shipment-notices")
    @RequirePermission(SupplyChainPermissions.PORTAL_SHIPMENT_CREATE)
    public SupplierShipmentNoticeDTO createShipmentNotice(@RequestBody SupplierShipmentNoticeDTO dto) {
        return portalService.createShipmentNotice(dto);
    }

    @PutMapping("/shipment-notices/{id}")
    @RequirePermission(SupplyChainPermissions.PORTAL_SHIPMENT_UPDATE)
    public SupplierShipmentNoticeDTO updateShipmentNotice(
            @PathVariable Long id,
            @RequestBody SupplierShipmentNoticeDTO dto) {
        return portalService.updateShipmentNotice(id, dto);
    }

    @PostMapping("/shipment-notices/{id}/submit")
    @RequirePermission(SupplyChainPermissions.PORTAL_SHIPMENT_SUBMIT)
    public SupplierShipmentNoticeDTO submitShipmentNotice(@PathVariable Long id) {
        return portalService.submitShipmentNotice(id);
    }

    @PostMapping("/shipment-notices/{id}/confirm")
    @RequirePermission(SupplyChainPermissions.PORTAL_SHIPMENT_CONFIRM)
    public SupplierShipmentNoticeDTO confirmShipmentNotice(
            @PathVariable Long id,
            @RequestParam String confirmer) {
        return portalService.confirmShipmentNotice(id, confirmer);
    }

    @GetMapping("/shipment-notices/{id}")
    @RequirePermission(SupplyChainPermissions.PORTAL_SHIPMENT_READ)
    public SupplierShipmentNoticeDTO queryShipmentNotice(@PathVariable Long id) {
        return portalService.queryShipmentNotice(id);
    }

    @GetMapping("/shipment-notices")
    @RequirePermission(SupplyChainPermissions.PORTAL_SHIPMENT_READ)
    public List<SupplierShipmentNoticeDTO> queryShipmentNotices(
            @RequestParam String supplierCode,
            @RequestParam(required = false) String status) {
        return portalService.queryShipmentNotices(supplierCode, status);
    }

    // ==================== 对账 ====================
    @PostMapping("/reconciliations")
    @RequirePermission(SupplyChainPermissions.PORTAL_RECONCILIATION_CREATE)
    public SupplierReconciliationDTO createReconciliation(@RequestBody SupplierReconciliationDTO dto) {
        return portalService.createReconciliation(dto);
    }

    @PostMapping("/reconciliations/{id}/submit")
    @RequirePermission(SupplyChainPermissions.PORTAL_RECONCILIATION_SUBMIT)
    public SupplierReconciliationDTO submitReconciliation(@PathVariable Long id) {
        return portalService.submitReconciliation(id);
    }

    @PostMapping("/reconciliations/{id}/confirm")
    @RequirePermission(SupplyChainPermissions.PORTAL_RECONCILIATION_CONFIRM)
    public SupplierReconciliationDTO confirmReconciliation(
            @PathVariable Long id,
            @RequestParam String confirmedBy) {
        return portalService.confirmReconciliation(id, confirmedBy);
    }

    @PostMapping("/reconciliations/{id}/reject")
    @RequirePermission(SupplyChainPermissions.PORTAL_RECONCILIATION_REJECT)
    public SupplierReconciliationDTO rejectReconciliation(
            @PathVariable Long id,
            @RequestParam String remark) {
        return portalService.rejectReconciliation(id, remark);
    }

    @PostMapping("/reconciliations/{id}/paid")
    @RequirePermission(SupplyChainPermissions.PORTAL_RECONCILIATION_PAID)
    public SupplierReconciliationDTO markAsPaid(@PathVariable Long id) {
        return portalService.markAsPaid(id);
    }

    @GetMapping("/reconciliations/{id}")
    @RequirePermission(SupplyChainPermissions.PORTAL_RECONCILIATION_READ)
    public SupplierReconciliationDTO queryReconciliation(@PathVariable Long id) {
        return portalService.queryReconciliation(id);
    }

    @GetMapping("/reconciliations")
    @RequirePermission(SupplyChainPermissions.PORTAL_RECONCILIATION_READ)
    public List<SupplierReconciliationDTO> queryReconciliations(
            @RequestParam String supplierCode,
            @RequestParam(required = false) String status) {
        return portalService.queryReconciliations(supplierCode, status);
    }
}