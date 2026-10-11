package com.metawebthree.supplier.interfaces.controller;

import com.metawebthree.supplier.application.dto.SupplierDTO;
import com.metawebthree.supplier.application.dto.SupplierPerformanceDTO;
import com.metawebthree.supplier.application.dto.SupplierRegistrationDTO;
import com.metawebthree.supplier.application.dto.SupplierVerificationDTO;
import com.metawebthree.supplier.application.service.SupplierApplicationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/supplier")
public class SupplierController {

    private final SupplierApplicationService supplierService;

    public SupplierController(SupplierApplicationService supplierService) {
        this.supplierService = supplierService;
    }

    @PostMapping("/register")
    public ResponseEntity<SupplierDTO> register(@RequestBody SupplierRegistrationDTO registrationDTO) {
        SupplierDTO result = supplierService.register(registrationDTO);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/{id}/submit-verification")
    public ResponseEntity<SupplierDTO> submitForVerification(@PathVariable Long id) {
        SupplierDTO result = supplierService.submitForVerification(id);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/verify")
    public ResponseEntity<SupplierDTO> verify(@RequestBody SupplierVerificationDTO verificationDTO) {
        SupplierDTO result = supplierService.verify(verificationDTO);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/{id}")
    public ResponseEntity<SupplierDTO> getSupplierById(@PathVariable Long id) {
        SupplierDTO result = supplierService.getSupplierById(id);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/code/{code}")
    public ResponseEntity<SupplierDTO> getSupplierByCode(@PathVariable String code) {
        SupplierDTO result = supplierService.getSupplierByCode(code);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/list")
    public ResponseEntity<List<SupplierDTO>> listAllSuppliers() {
        List<SupplierDTO> result = supplierService.listAllSuppliers();
        return ResponseEntity.ok(result);
    }

    @GetMapping("/list/by-status")
    public ResponseEntity<List<SupplierDTO>> listSuppliersByStatus(@RequestParam Integer status) {
        List<SupplierDTO> result = supplierService.listSuppliersByStatus(status);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/list/by-verification-status")
    public ResponseEntity<List<SupplierDTO>> listSuppliersByVerificationStatus(@RequestParam Integer verificationStatus) {
        List<SupplierDTO> result = supplierService.listSuppliersByVerificationStatus(verificationStatus);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/{id}/performance")
    public ResponseEntity<SupplierPerformanceDTO> evaluatePerformance(@PathVariable Long id) {
        SupplierPerformanceDTO result = supplierService.evaluatePerformance(id);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/{id}/score")
    public ResponseEntity<SupplierDTO> updateScore(@PathVariable Long id, @RequestParam Integer delta) {
        SupplierDTO result = supplierService.updateScore(id, delta);
        return ResponseEntity.ok(result);
    }
}