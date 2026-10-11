package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.entity.QcInspectionItem;
import com.metawebthree.mes.domain.repository.QcInspectionItemRepository;
import com.metawebthree.mes.interfaces.dto.QcInspectionItemDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/qc/inspection-item")
public class QcInspectionItemController {
    
    private final QcInspectionItemRepository repository;
    
    public QcInspectionItemController(QcInspectionItemRepository repository) {
        this.repository = repository;
    }
    
    @GetMapping
    @RequirePermission(MesPermissions.QC_INSPECTION_ITEM_READ)
    public ResponseEntity<List<QcInspectionItemDTO>> getAll() {
        List<QcInspectionItemDTO> dtos = repository.findAll().stream()
                .map(QcInspectionItemDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/{id}")
    @RequirePermission(MesPermissions.QC_INSPECTION_ITEM_READ)
    public ResponseEntity<QcInspectionItemDTO> getById(@PathVariable Long id) {
        return repository.findById(id)
                .map(item -> ResponseEntity.ok(QcInspectionItemDTO.fromEntity(item)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/category/{itemCategory}")
    @RequirePermission(MesPermissions.QC_INSPECTION_ITEM_READ)
    public ResponseEntity<List<QcInspectionItemDTO>> getByCategory(@PathVariable String itemCategory) {
        List<QcInspectionItemDTO> dtos = repository.findByItemCategory(itemCategory).stream()
                .map(QcInspectionItemDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/status/{status}")
    @RequirePermission(MesPermissions.QC_INSPECTION_ITEM_READ)
    public ResponseEntity<List<QcInspectionItemDTO>> getByStatus(@PathVariable String status) {
        QcInspectionItem.ItemStatus itemStatus = Enum.valueOf(QcInspectionItem.ItemStatus.class, status);
        List<QcInspectionItemDTO> dtos = repository.findByStatus(itemStatus).stream()
                .map(QcInspectionItemDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping
    @RequirePermission(MesPermissions.QC_INSPECTION_ITEM_CREATE)
    public ResponseEntity<QcInspectionItemDTO> create(@RequestBody Map<String, Object> request) {
        String itemCode = (String) request.get("itemCode");
        String itemName = (String) request.get("itemName");
        String itemCategory = (String) request.get("itemCategory");
        
        if (repository.existsByItemCode(itemCode)) {
            return ResponseEntity.badRequest().build();
        }
        
        QcInspectionItem entity = QcInspectionItem.create(itemCode, itemName, itemCategory);
        
        if (request.containsKey("dataType")) {
            entity.setDataType((String) request.get("dataType"));
        }
        if (request.containsKey("unit")) {
            entity.setUnit((String) request.get("unit"));
        }
        if (request.containsKey("standardValue")) {
            entity.setStandardValue(((Number) request.get("standardValue")).doubleValue());
        }
        if (request.containsKey("upperLimit")) {
            entity.setUpperLimit(((Number) request.get("upperLimit")).doubleValue());
        }
        if (request.containsKey("lowerLimit")) {
            entity.setLowerLimit(((Number) request.get("lowerLimit")).doubleValue());
        }
        if (request.containsKey("inspectionMethod")) {
            entity.setInspectionMethod((String) request.get("inspectionMethod"));
        }
        if (request.containsKey("inspectionTool")) {
            entity.setInspectionTool((String) request.get("inspectionTool"));
        }
        if (request.containsKey("severity")) {
            entity.setSeverity((Integer) request.get("severity"));
        }
        if (request.containsKey("isMandatory")) {
            entity.setIsMandatory((Boolean) request.get("isMandatory"));
        }
        if (request.containsKey("sortOrder")) {
            entity.setSortOrder((Integer) request.get("sortOrder"));
        }
        if (request.containsKey("remark")) {
            entity.setRemark((String) request.get("remark"));
        }
        
        QcInspectionItem saved = repository.save(entity);
        return ResponseEntity.status(201).body(QcInspectionItemDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    @RequirePermission(MesPermissions.QC_INSPECTION_ITEM_UPDATE)
    public ResponseEntity<QcInspectionItemDTO> update(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        
        return repository.findById(id)
                .map(entity -> {
                    if (request.containsKey("itemName")) {
                        entity.setItemName((String) request.get("itemName"));
                    }
                    if (request.containsKey("itemCategory")) {
                        entity.setItemCategory((String) request.get("itemCategory"));
                    }
                    if (request.containsKey("dataType")) {
                        entity.setDataType((String) request.get("dataType"));
                    }
                    if (request.containsKey("unit")) {
                        entity.setUnit((String) request.get("unit"));
                    }
                    if (request.containsKey("standardValue")) {
                        entity.setStandardValue(((Number) request.get("standardValue")).doubleValue());
                    }
                    if (request.containsKey("upperLimit")) {
                        entity.setUpperLimit(((Number) request.get("upperLimit")).doubleValue());
                    }
                    if (request.containsKey("lowerLimit")) {
                        entity.setLowerLimit(((Number) request.get("lowerLimit")).doubleValue());
                    }
                    if (request.containsKey("inspectionMethod")) {
                        entity.setInspectionMethod((String) request.get("inspectionMethod"));
                    }
                    if (request.containsKey("inspectionTool")) {
                        entity.setInspectionTool((String) request.get("inspectionTool"));
                    }
                    if (request.containsKey("severity")) {
                        entity.setSeverity((Integer) request.get("severity"));
                    }
                    if (request.containsKey("isMandatory")) {
                        entity.setIsMandatory((Boolean) request.get("isMandatory"));
                    }
                    if (request.containsKey("sortOrder")) {
                        entity.setSortOrder((Integer) request.get("sortOrder"));
                    }
                    if (request.containsKey("remark")) {
                        entity.setRemark((String) request.get("remark"));
                    }
                    
                    QcInspectionItem saved = repository.save(entity);
                    return ResponseEntity.ok(QcInspectionItemDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @DeleteMapping("/{id}")
    @RequirePermission(MesPermissions.QC_INSPECTION_ITEM_DELETE)
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        if (repository.findById(id).isPresent()) {
            repository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
    
    @PostMapping("/{id}/activate")
    @RequirePermission(MesPermissions.QC_INSPECTION_ITEM_UPDATE)
    public ResponseEntity<QcInspectionItemDTO> activate(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> {
                    entity.activate();
                    QcInspectionItem saved = repository.save(entity);
                    return ResponseEntity.ok(QcInspectionItemDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/{id}/deactivate")
    @RequirePermission(MesPermissions.QC_INSPECTION_ITEM_UPDATE)
    public ResponseEntity<QcInspectionItemDTO> deactivate(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> {
                    entity.deactivate();
                    QcInspectionItem saved = repository.save(entity);
                    return ResponseEntity.ok(QcInspectionItemDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
}