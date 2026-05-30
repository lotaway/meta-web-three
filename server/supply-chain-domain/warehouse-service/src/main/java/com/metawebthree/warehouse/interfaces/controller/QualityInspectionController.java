package com.metawebthree.warehouse.interfaces.controller;

import com.metawebthree.warehouse.application.QualityInspectionApplicationService;
import com.metawebthree.warehouse.application.dto.QualityInspectionDTO;
import com.metawebthree.warehouse.application.dto.QualityStandardDTO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/quality")
public class QualityInspectionController {
    
    @Autowired
    private QualityInspectionApplicationService service;
    
    // Quality Standard APIs
    @PostMapping("/standards")
    public ResponseEntity<QualityStandardDTO> createStandard(@RequestBody QualityStandardDTO dto) {
        QualityStandardDTO result = service.createStandard(dto);
        return ResponseEntity.ok(result);
    }
    
    @GetMapping("/standards/{skuCode}")
    public ResponseEntity<QualityStandardDTO> getStandardBySku(@PathVariable String skuCode) {
        QualityStandardDTO result = service.getStandardBySku(skuCode);
        if (result == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(result);
    }
    
    @GetMapping("/standards")
    public ResponseEntity<List<QualityStandardDTO>> listStandards(
            @RequestParam(required = false) Boolean activeOnly) {
        List<QualityStandardDTO> list = service.listStandards(activeOnly);
        return ResponseEntity.ok(list);
    }
    
    // Quality Inspection APIs
    @PostMapping("/inspections")
    public ResponseEntity<QualityInspectionDTO> createInspection(@RequestBody QualityInspectionDTO dto) {
        QualityInspectionDTO result = service.createInspection(dto);
        return ResponseEntity.ok(result);
    }
    
    @GetMapping("/inspections/{id}")
    public ResponseEntity<QualityInspectionDTO> getInspectionById(@PathVariable Long id) {
        QualityInspectionDTO result = service.getInspectionById(id);
        if (result == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(result);
    }
    
    @GetMapping("/inspections/no/{inspectionNo}")
    public ResponseEntity<QualityInspectionDTO> getInspectionByNo(@PathVariable String inspectionNo) {
        QualityInspectionDTO result = service.getInspectionByNo(inspectionNo);
        if (result == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(result);
    }
    
    @GetMapping("/inspections")
    public ResponseEntity<List<QualityInspectionDTO>> listInspections() {
        List<QualityInspectionDTO> list = service.listInspections();
        return ResponseEntity.ok(list);
    }
    
    // Health check
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        Map<String, String> result = new HashMap<>();
        result.put("status", "UP");
        result.put("module", "quality-inspection");
        return ResponseEntity.ok(result);
    }
}