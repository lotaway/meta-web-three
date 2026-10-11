package com.metawebthree.supplier.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.SupplyChainPermissions;
import com.metawebthree.supplier.application.SupplierApplicationService;
import com.metawebthree.supplier.application.dto.SupplierDTO;
import org.springframework.web.bind.annotation.*;
import java.util.List;

/**
 * 供应商管理 REST API
 * 能力: 供应商档案、资质管理、评估分级
 */
@RestController
@RequestMapping("/api/supplier")
public class SupplierController {

    private final SupplierApplicationService supplierService;

    public SupplierController(SupplierApplicationService supplierService) {
        this.supplierService = supplierService;
    }

    @PostMapping("/suppliers")
    @RequirePermission(SupplyChainPermissions.SUPPLIER_CREATE)
    public SupplierDTO createSupplier(@RequestBody SupplierDTO dto) {
        return supplierService.createSupplier(dto);
    }

    @PutMapping("/suppliers/{id}")
    @RequirePermission(SupplyChainPermissions.SUPPLIER_UPDATE)
    public SupplierDTO updateSupplier(
            @PathVariable Long id,
            @RequestBody SupplierDTO dto) {
        return supplierService.updateSupplier(id, dto);
    }

    @GetMapping("/suppliers/{id}")
    @RequirePermission(SupplyChainPermissions.SUPPLIER_READ)
    public SupplierDTO querySupplier(@PathVariable Long id) {
        return supplierService.querySupplier(id);
    }

    @GetMapping("/suppliers/code/{code}")
    @RequirePermission(SupplyChainPermissions.SUPPLIER_READ)
    public SupplierDTO queryByCode(@PathVariable String code) {
        return supplierService.queryByCode(code);
    }

    @GetMapping("/suppliers")
    @RequirePermission(SupplyChainPermissions.SUPPLIER_READ)
    public List<SupplierDTO> listSuppliers(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String category) {
        return supplierService.listSuppliers(status, category);
    }

    @PutMapping("/suppliers/{id}/assessment")
    @RequirePermission(SupplyChainPermissions.SUPPLIER_ASSESS)
    public SupplierDTO updateAssessment(
            @PathVariable Long id,
            @RequestParam String level) {
        return supplierService.updateAssessment(id, level);
    }
}