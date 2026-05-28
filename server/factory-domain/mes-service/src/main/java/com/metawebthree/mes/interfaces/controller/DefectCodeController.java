package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.entity.DefectCode;
import com.metawebthree.mes.domain.repository.DefectCodeRepository;
import com.metawebthree.mes.interfaces.dto.DefectCodeDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/qc/defect-code")
public class DefectCodeController {
    
    private final DefectCodeRepository defectCodeRepository;
    
    public DefectCodeController(DefectCodeRepository defectCodeRepository) {
        this.defectCodeRepository = defectCodeRepository;
    }
    
    @GetMapping
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_READ)
    public ResponseEntity<List<DefectCodeDTO>> getAll() {
        List<DefectCodeDTO> dtos = defectCodeRepository.findAll().stream()
                .map(DefectCodeDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/{id}")
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_READ)
    public ResponseEntity<DefectCodeDTO> getById(@PathVariable Long id) {
        return defectCodeRepository.findById(id)
                .map(entity -> ResponseEntity.ok(DefectCodeDTO.fromEntity(entity)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code/{code}")
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_READ)
    public ResponseEntity<DefectCodeDTO> getByCode(@PathVariable String code) {
        return defectCodeRepository.findByDefectCode(code)
                .map(entity -> ResponseEntity.ok(DefectCodeDTO.fromEntity(entity)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/category/{category}")
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_READ)
    public ResponseEntity<List<DefectCodeDTO>> getByCategory(@PathVariable String category) {
        DefectCode.DefectCategory cat = DefectCode.DefectCategory.valueOf(category);
        List<DefectCodeDTO> dtos = defectCodeRepository.findByCategory(cat).stream()
                .map(DefectCodeDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/severity/{severity}")
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_READ)
    public ResponseEntity<List<DefectCodeDTO>> getBySeverity(@PathVariable String severity) {
        DefectCode.DefectSeverity sev = DefectCode.DefectSeverity.valueOf(severity);
        List<DefectCodeDTO> dtos = defectCodeRepository.findBySeverity(sev).stream()
                .map(DefectCodeDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/enabled")
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_READ)
    public ResponseEntity<List<DefectCodeDTO>> getEnabled() {
        List<DefectCodeDTO> dtos = defectCodeRepository.findByIsEnabled(true).stream()
                .map(DefectCodeDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_CREATE)
    public ResponseEntity<DefectCodeDTO> create(@RequestBody Map<String, Object> request) {
        String defectCode = (String) request.get("defectCode");
        String defectName = (String) request.get("defectName");
        String category = (String) request.get("category");
        String severity = (String) request.get("severity");
        
        if (defectCodeRepository.existsByDefectCode(defectCode)) {
            return ResponseEntity.badRequest().build();
        }
        
        DefectCode entity = new DefectCode();
        entity.create(defectCode, defectName, 
                DefectCode.DefectCategory.valueOf(category),
                DefectCode.DefectSeverity.valueOf(severity));
        
        if (request.containsKey("description")) {
            entity.setDescription((String) request.get("description"));
        }
        if (request.containsKey("dispositionGuide")) {
            entity.setDispositionGuide((String) request.get("dispositionGuide"));
        }
        
        DefectCode saved = defectCodeRepository.save(entity);
        return ResponseEntity.status(201).body(DefectCodeDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_UPDATE)
    public ResponseEntity<DefectCodeDTO> update(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        
        return defectCodeRepository.findById(id)
                .map(entity -> {
                    if (request.containsKey("defectName")) {
                        entity.updateName((String) request.get("defectName"));
                    }
                    if (request.containsKey("category")) {
                        entity.updateCategory(DefectCode.DefectCategory.valueOf((String) request.get("category")));
                    }
                    if (request.containsKey("severity")) {
                        entity.updateSeverity(DefectCode.DefectSeverity.valueOf((String) request.get("severity")));
                    }
                    if (request.containsKey("description")) {
                        entity.setDescription((String) request.get("description"));
                    }
                    if (request.containsKey("dispositionGuide")) {
                        entity.updateDispositionGuide((String) request.get("dispositionGuide"));
                    }
                    if (request.containsKey("sortOrder")) {
                        entity.updateSortOrder((Integer) request.get("sortOrder"));
                    }
                    
                    DefectCode saved = defectCodeRepository.save(entity);
                    return ResponseEntity.ok(DefectCodeDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @DeleteMapping("/{id}")
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_DELETE)
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        if (defectCodeRepository.findById(id).isPresent()) {
            defectCodeRepository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
    
    @PostMapping("/{id}/enable")
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_UPDATE)
    public ResponseEntity<DefectCodeDTO> enable(@PathVariable Long id) {
        return defectCodeRepository.findById(id)
                .map(entity -> {
                    entity.enable();
                    DefectCode saved = defectCodeRepository.save(entity);
                    return ResponseEntity.ok(DefectCodeDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/{id}/disable")
    @RequirePermission(MesPermissions.QC_DEFECT_CODE_UPDATE)
    public ResponseEntity<DefectCodeDTO> disable(@PathVariable Long id) {
        return defectCodeRepository.findById(id)
                .map(entity -> {
                    entity.disable();
                    DefectCode saved = defectCodeRepository.save(entity);
                    return ResponseEntity.ok(DefectCodeDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
}