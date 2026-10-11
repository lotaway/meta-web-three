package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.entity.QcInspectionType;
import com.metawebthree.mes.domain.repository.QcInspectionTypeRepository;
import com.metawebthree.mes.interfaces.dto.QcInspectionTypeDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/qc/inspection-type")
public class QcInspectionTypeController {
    
    private final QcInspectionTypeRepository repository;
    
    public QcInspectionTypeController(QcInspectionTypeRepository repository) {
        this.repository = repository;
    }
    
    @GetMapping
    @RequirePermission(MesPermissions.QC_INSPECTION_TYPE_READ)
    public ResponseEntity<List<QcInspectionTypeDTO>> getAll() {
        List<QcInspectionTypeDTO> dtos = repository.findAll().stream()
                .map(QcInspectionTypeDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/{id}")
    @RequirePermission(MesPermissions.QC_INSPECTION_TYPE_READ)
    public ResponseEntity<QcInspectionTypeDTO> getById(@PathVariable Long id) {
        return repository.findById(id)
                .map(type -> ResponseEntity.ok(QcInspectionTypeDTO.fromEntity(type)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/category/{category}")
    @RequirePermission(MesPermissions.QC_INSPECTION_TYPE_READ)
    public ResponseEntity<List<QcInspectionTypeDTO>> getByCategory(@PathVariable String category) {
        QcInspectionType.InspectionCategory cat = QcInspectionType.InspectionCategory.valueOf(category);
        List<QcInspectionTypeDTO> dtos = repository.findByCategory(cat).stream()
                .map(QcInspectionTypeDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping
    @RequirePermission(MesPermissions.QC_INSPECTION_TYPE_CREATE)
    public ResponseEntity<QcInspectionTypeDTO> create(@RequestBody Map<String, Object> request) {
        String typeCode = (String) request.get("typeCode");
        String typeName = (String) request.get("typeName");
        String categoryStr = (String) request.get("category");
        
        if (repository.existsByTypeCode(typeCode)) {
            return ResponseEntity.badRequest().build();
        }
        
        QcInspectionType.InspectionCategory category = QcInspectionType.InspectionCategory.valueOf(categoryStr);
        QcInspectionType entity = QcInspectionType.create(typeCode, typeName, category);
        
        if (request.containsKey("description")) {
            entity.setDescription((String) request.get("description"));
        }
        if (request.containsKey("applicableProducts")) {
            entity.setApplicableProducts((String) request.get("applicableProducts"));
        }
        if (request.containsKey("defaultSamplingPlan")) {
            entity.setDefaultSamplingPlan((String) request.get("defaultSamplingPlan"));
        }
        if (request.containsKey("defaultAql")) {
            entity.setDefaultAql((String) request.get("defaultAql"));
        }
        if (request.containsKey("defaultTimeoutHours")) {
            entity.setDefaultTimeoutHours((Integer) request.get("defaultTimeoutHours"));
        }
        if (request.containsKey("requireCertificate")) {
            entity.setRequireCertificate((Boolean) request.get("requireCertificate"));
        }
        if (request.containsKey("requireTestReport")) {
            entity.setRequireTestReport((Boolean) request.get("requireTestReport"));
        }
        if (request.containsKey("sortOrder")) {
            entity.setSortOrder((Integer) request.get("sortOrder"));
        }
        
        QcInspectionType saved = repository.save(entity);
        return ResponseEntity.status(201).body(QcInspectionTypeDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    @RequirePermission(MesPermissions.QC_INSPECTION_TYPE_UPDATE)
    public ResponseEntity<QcInspectionTypeDTO> update(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        
        return repository.findById(id)
                .map(entity -> {
                    if (request.containsKey("typeName")) {
                        entity.setTypeName((String) request.get("typeName"));
                    }
                    if (request.containsKey("category")) {
                        entity.setCategory(QcInspectionType.InspectionCategory.valueOf((String) request.get("category")));
                    }
                    if (request.containsKey("description")) {
                        entity.setDescription((String) request.get("description"));
                    }
                    if (request.containsKey("applicableProducts")) {
                        entity.setApplicableProducts((String) request.get("applicableProducts"));
                    }
                    if (request.containsKey("defaultSamplingPlan")) {
                        entity.setDefaultSamplingPlan((String) request.get("defaultSamplingPlan"));
                    }
                    if (request.containsKey("defaultAql")) {
                        entity.setDefaultAql((String) request.get("defaultAql"));
                    }
                    if (request.containsKey("defaultTimeoutHours")) {
                        entity.setDefaultTimeoutHours((Integer) request.get("defaultTimeoutHours"));
                    }
                    if (request.containsKey("requireCertificate")) {
                        entity.setRequireCertificate((Boolean) request.get("requireCertificate"));
                    }
                    if (request.containsKey("requireTestReport")) {
                        entity.setRequireTestReport((Boolean) request.get("requireTestReport"));
                    }
                    if (request.containsKey("sortOrder")) {
                        entity.setSortOrder((Integer) request.get("sortOrder"));
                    }
                    
                    QcInspectionType saved = repository.save(entity);
                    return ResponseEntity.ok(QcInspectionTypeDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @DeleteMapping("/{id}")
    @RequirePermission(MesPermissions.QC_INSPECTION_TYPE_DELETE)
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        if (repository.findById(id).isPresent()) {
            repository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
    
    @PostMapping("/{id}/activate")
    @RequirePermission(MesPermissions.QC_INSPECTION_TYPE_UPDATE)
    public ResponseEntity<QcInspectionTypeDTO> activate(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> {
                    entity.activate();
                    QcInspectionType saved = repository.save(entity);
                    return ResponseEntity.ok(QcInspectionTypeDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/{id}/deactivate")
    @RequirePermission(MesPermissions.QC_INSPECTION_TYPE_UPDATE)
    public ResponseEntity<QcInspectionTypeDTO> deactivate(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> {
                    entity.deactivate();
                    QcInspectionType saved = repository.save(entity);
                    return ResponseEntity.ok(QcInspectionTypeDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
}