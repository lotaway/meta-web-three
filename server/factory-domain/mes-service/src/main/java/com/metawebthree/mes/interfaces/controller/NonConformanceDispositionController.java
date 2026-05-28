package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.entity.NonConformanceDisposition;
import com.metawebthree.mes.domain.entity.NonConformanceDisposition.DispositionType;
import com.metawebthree.mes.domain.repository.NonConformanceDispositionRepository;
import com.metawebthree.mes.interfaces.dto.NonConformanceDispositionDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/qc/non-conformance")
public class NonConformanceDispositionController {
    
    private final NonConformanceDispositionRepository repository;
    
    public NonConformanceDispositionController(NonConformanceDispositionRepository repository) {
        this.repository = repository;
    }
    
    @GetMapping
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_READ)
    public ResponseEntity<List<NonConformanceDispositionDTO>> getAll() {
        List<NonConformanceDispositionDTO> dtos = repository.findAll().stream()
                .map(NonConformanceDispositionDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/{id}")
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_READ)
    public ResponseEntity<NonConformanceDispositionDTO> getById(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> ResponseEntity.ok(NonConformanceDispositionDTO.fromEntity(entity)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code/{code}")
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_READ)
    public ResponseEntity<NonConformanceDispositionDTO> getByCode(@PathVariable String code) {
        return repository.findByDispositionCode(code)
                .map(entity -> ResponseEntity.ok(NonConformanceDispositionDTO.fromEntity(entity)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/type/{type}")
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_READ)
    public ResponseEntity<List<NonConformanceDispositionDTO>> getByType(@PathVariable String type) {
        DispositionType dispositionType = DispositionType.valueOf(type);
        List<NonConformanceDispositionDTO> dtos = repository.findByType(dispositionType).stream()
                .map(NonConformanceDispositionDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/enabled")
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_READ)
    public ResponseEntity<List<NonConformanceDispositionDTO>> getEnabled() {
        List<NonConformanceDispositionDTO> dtos = repository.findByIsEnabled(true).stream()
                .map(NonConformanceDispositionDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_CREATE)
    public ResponseEntity<NonConformanceDispositionDTO> create(@RequestBody Map<String, Object> request) {
        String dispositionCode = (String) request.get("dispositionCode");
        String dispositionName = (String) request.get("dispositionName");
        String typeStr = (String) request.get("type");
        
        if (repository.existsByDispositionCode(dispositionCode)) {
            return ResponseEntity.badRequest().build();
        }
        
        NonConformanceDisposition entity = new NonConformanceDisposition();
        entity.create(dispositionCode, dispositionName, DispositionType.valueOf(typeStr));
        
        if (request.containsKey("sortOrder")) {
            entity.setSortOrder((Integer) request.get("sortOrder"));
        }
        
        if (request.containsKey("defaultFlow")) {
            String flowType = (String) request.get("defaultFlow");
            if ("SCRAP".equals(flowType)) {
                entity.addDefaultScrapFlow();
            } else if ("REWORK".equals(flowType)) {
                entity.addDefaultReworkFlow();
            }
        }
        
        NonConformanceDisposition saved = repository.save(entity);
        return ResponseEntity.status(201).body(NonConformanceDispositionDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_UPDATE)
    public ResponseEntity<NonConformanceDispositionDTO> update(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        
        return repository.findById(id)
                .map(entity -> {
                    if (request.containsKey("dispositionName")) {
                        entity.setDispositionName((String) request.get("dispositionName"));
                    }
                    if (request.containsKey("sortOrder")) {
                        entity.updateSortOrder((Integer) request.get("sortOrder"));
                    }
                    
                    NonConformanceDisposition saved = repository.save(entity);
                    return ResponseEntity.ok(NonConformanceDispositionDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @DeleteMapping("/{id}")
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_DELETE)
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        if (repository.findById(id).isPresent()) {
            repository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
    
    @PostMapping("/{id}/enable")
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_UPDATE)
    public ResponseEntity<NonConformanceDispositionDTO> enable(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> {
                    entity.enable();
                    NonConformanceDisposition saved = repository.save(entity);
                    return ResponseEntity.ok(NonConformanceDispositionDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/{id}/disable")
    @RequirePermission(MesPermissions.QC_NON_CONFORMANCE_UPDATE)
    public ResponseEntity<NonConformanceDispositionDTO> disable(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> {
                    entity.disable();
                    NonConformanceDisposition saved = repository.save(entity);
                    return ResponseEntity.ok(NonConformanceDispositionDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
}