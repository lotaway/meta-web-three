package com.metawebthree.supplier.interfaces.controller;

import com.metawebthree.supplier.application.dto.SupplierPerformanceDTO;
import com.metawebthree.supplier.application.SupplierPerformanceApplicationService;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/supplier-performance")
public class SupplierPerformanceController {
    
    private final SupplierPerformanceApplicationService supplierPerformanceApplicationService;
    
    public SupplierPerformanceController(SupplierPerformanceApplicationService supplierPerformanceApplicationService) {
        this.supplierPerformanceApplicationService = supplierPerformanceApplicationService;
    }
    
    @PostMapping
    public SupplierPerformanceDTO createOrUpdateEvaluation(@RequestBody SupplierPerformanceDTO dto) {
        return supplierPerformanceApplicationService.createOrUpdateEvaluation(dto);
    }
    
    @GetMapping("/{id}")
    public SupplierPerformanceDTO getById(@PathVariable Long id) {
        return supplierPerformanceApplicationService.getById(id);
    }
    
    @GetMapping("/supplier/{supplierId}")
    public List<SupplierPerformanceDTO> getBySupplierId(@PathVariable Long supplierId) {
        return supplierPerformanceApplicationService.getBySupplierId(supplierId);
    }
    
    @GetMapping
    public List<SupplierPerformanceDTO> getAll() {
        return supplierPerformanceApplicationService.getAll();
    }
    
    @GetMapping("/level/{level}")
    public List<SupplierPerformanceDTO> getByAssessmentLevel(@PathVariable String level) {
        return supplierPerformanceApplicationService.getByAssessmentLevel(level);
    }
    
    @DeleteMapping("/{id}")
    public void deleteById(@PathVariable Long id) {
        supplierPerformanceApplicationService.deleteById(id);
    }
    
    @GetMapping("/dashboard")
    public Map<String, Object> getDashboard() {
        return supplierPerformanceApplicationService.getDashboard();
    }
}