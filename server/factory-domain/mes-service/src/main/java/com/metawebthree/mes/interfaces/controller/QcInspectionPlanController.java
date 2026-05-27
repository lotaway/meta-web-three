package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.entity.QcInspectionPlan;
import com.metawebthree.mes.domain.repository.QcInspectionPlanRepository;
import com.metawebthree.mes.domain.repository.QcPlanItemRepository;
import com.metawebthree.mes.domain.repository.QcInspectionItemRepository;
import com.metawebthree.mes.interfaces.dto.QcInspectionPlanDTO;
import com.metawebthree.mes.interfaces.dto.QcPlanItemDTO;
import com.metawebthree.mes.interfaces.dto.QcInspectionItemDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/qc/inspection-plan")
public class QcInspectionPlanController {
    
    private final QcInspectionPlanRepository planRepository;
    private final QcPlanItemRepository planItemRepository;
    private final QcInspectionItemRepository itemRepository;
    
    public QcInspectionPlanController(
            QcInspectionPlanRepository planRepository,
            QcPlanItemRepository planItemRepository,
            QcInspectionItemRepository itemRepository) {
        this.planRepository = planRepository;
        this.planItemRepository = planItemRepository;
        this.itemRepository = itemRepository;
    }
    
    @GetMapping
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_READ)
    public ResponseEntity<List<QcInspectionPlanDTO>> getAll() {
        List<QcInspectionPlanDTO> dtos = planRepository.findAll().stream()
                .map(QcInspectionPlanDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/{id}")
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_READ)
    public ResponseEntity<QcInspectionPlanDTO> getById(@PathVariable Long id) {
        return planRepository.findById(id)
                .map(plan -> {
                    QcInspectionPlanDTO dto = QcInspectionPlanDTO.fromEntity(plan);
                    List<QcPlanItemDTO> items = planItemRepository.findByPlanId(id).stream()
                            .map(item -> {
                                QcPlanItemDTO itemDto = QcPlanItemDTO.fromEntity(item);
                                itemRepository.findById(item.getItemId())
                                        .ifPresent(inspectionItem -> 
                                            itemDto.setInspectionItem(QcInspectionItemDTO.fromEntity(inspectionItem)));
                                return itemDto;
                            })
                            .collect(Collectors.toList());
                    dto.setPlanItems(items);
                    return ResponseEntity.ok(dto);
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/type/{inspectionType}")
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_READ)
    public ResponseEntity<List<QcInspectionPlanDTO>> getByInspectionType(@PathVariable String inspectionType) {
        List<QcInspectionPlanDTO> dtos = planRepository.findByInspectionType(inspectionType).stream()
                .map(QcInspectionPlanDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/status/{status}")
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_READ)
    public ResponseEntity<List<QcInspectionPlanDTO>> getByStatus(@PathVariable String status) {
        QcInspectionPlan.PlanStatus planStatus = QcInspectionPlan.PlanStatus.valueOf(status);
        List<QcInspectionPlanDTO> dtos = planRepository.findByStatus(planStatus).stream()
                .map(QcInspectionPlanDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_CREATE)
    public ResponseEntity<QcInspectionPlanDTO> create(@RequestBody Map<String, Object> request) {
        String planCode = (String) request.get("planCode");
        String planName = (String) request.get("planName");
        String inspectionType = (String) request.get("inspectionType");
        
        if (planRepository.existsByPlanCode(planCode)) {
            return ResponseEntity.badRequest().build();
        }
        
        QcInspectionPlan entity = QcInspectionPlan.create(planCode, planName, inspectionType);
        
        if (request.containsKey("applicableProductTypes")) {
            entity.setApplicableProductTypes((String) request.get("applicableProductTypes"));
        }
        if (request.containsKey("samplingType")) {
            entity.setSamplingType((String) request.get("samplingType"));
        }
        if (request.containsKey("aql")) {
            entity.setAql((String) request.get("aql"));
        }
        if (request.containsKey("inspectionLevel")) {
            entity.setInspectionLevel((String) request.get("inspectionLevel"));
        }
        if (request.containsKey("sampleSize")) {
            entity.setSampleSize((Integer) request.get("sampleSize"));
        }
        if (request.containsKey("acceptNumber")) {
            entity.setAcceptNumber((String) request.get("acceptNumber"));
        }
        if (request.containsKey("rejectNumber")) {
            entity.setRejectNumber((String) request.get("rejectNumber"));
        }
        if (request.containsKey("dispositionRule")) {
            entity.setDispositionRule((String) request.get("dispositionRule"));
        }
        if (request.containsKey("qualifiedFlow")) {
            entity.setQualifiedFlow((String) request.get("qualifiedFlow"));
        }
        if (request.containsKey("unqualifiedFlow")) {
            entity.setUnqualifiedFlow((String) request.get("unqualifiedFlow"));
        }
        if (request.containsKey("specialApprovalFlow")) {
            entity.setSpecialApprovalFlow((String) request.get("specialApprovalFlow"));
        }
        if (request.containsKey("effectiveDate")) {
            entity.setEffectiveDate(LocalDateTime.parse((String) request.get("effectiveDate")));
        }
        if (request.containsKey("expirationDate")) {
            entity.setExpirationDate(LocalDateTime.parse((String) request.get("expirationDate")));
        }
        if (request.containsKey("sortOrder")) {
            entity.setSortOrder((Integer) request.get("sortOrder"));
        }
        if (request.containsKey("remark")) {
            entity.setRemark((String) request.get("remark"));
        }
        
        QcInspectionPlan saved = planRepository.save(entity);
        return ResponseEntity.status(201).body(QcInspectionPlanDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_UPDATE)
    public ResponseEntity<QcInspectionPlanDTO> update(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        
        return planRepository.findById(id)
                .map(entity -> {
                    if (request.containsKey("planName")) {
                        entity.setPlanName((String) request.get("planName"));
                    }
                    if (request.containsKey("inspectionType")) {
                        entity.setInspectionType((String) request.get("inspectionType"));
                    }
                    if (request.containsKey("applicableProductTypes")) {
                        entity.setApplicableProductTypes((String) request.get("applicableProductTypes"));
                    }
                    if (request.containsKey("samplingType")) {
                        entity.setSamplingType((String) request.get("samplingType"));
                    }
                    if (request.containsKey("aql")) {
                        entity.setAql((String) request.get("aql"));
                    }
                    if (request.containsKey("inspectionLevel")) {
                        entity.setInspectionLevel((String) request.get("inspectionLevel"));
                    }
                    if (request.containsKey("sampleSize")) {
                        entity.setSampleSize((Integer) request.get("sampleSize"));
                    }
                    if (request.containsKey("acceptNumber")) {
                        entity.setAcceptNumber((String) request.get("acceptNumber"));
                    }
                    if (request.containsKey("rejectNumber")) {
                        entity.setRejectNumber((String) request.get("rejectNumber"));
                    }
                    if (request.containsKey("dispositionRule")) {
                        entity.setDispositionRule((String) request.get("dispositionRule"));
                    }
                    if (request.containsKey("qualifiedFlow")) {
                        entity.setQualifiedFlow((String) request.get("qualifiedFlow"));
                    }
                    if (request.containsKey("unqualifiedFlow")) {
                        entity.setUnqualifiedFlow((String) request.get("unqualifiedFlow"));
                    }
                    if (request.containsKey("specialApprovalFlow")) {
                        entity.setSpecialApprovalFlow((String) request.get("specialApprovalFlow"));
                    }
                    if (request.containsKey("effectiveDate")) {
                        entity.setEffectiveDate(LocalDateTime.parse((String) request.get("effectiveDate")));
                    }
                    if (request.containsKey("expirationDate")) {
                        entity.setExpirationDate(LocalDateTime.parse((String) request.get("expirationDate")));
                    }
                    if (request.containsKey("sortOrder")) {
                        entity.setSortOrder((Integer) request.get("sortOrder"));
                    }
                    if (request.containsKey("remark")) {
                        entity.setRemark((String) request.get("remark"));
                    }
                    
                    QcInspectionPlan saved = planRepository.save(entity);
                    return ResponseEntity.ok(QcInspectionPlanDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @DeleteMapping("/{id}")
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_DELETE)
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        if (planRepository.findById(id).isPresent()) {
            planItemRepository.deleteByPlanId(id);
            planRepository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
    
    @PostMapping("/{id}/activate")
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_UPDATE)
    public ResponseEntity<QcInspectionPlanDTO> activate(@PathVariable Long id) {
        return planRepository.findById(id)
                .map(entity -> {
                    entity.activate();
                    QcInspectionPlan saved = planRepository.save(entity);
                    return ResponseEntity.ok(QcInspectionPlanDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/{id}/deactivate")
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_UPDATE)
    public ResponseEntity<QcInspectionPlanDTO> deactivate(@PathVariable Long id) {
        return planRepository.findById(id)
                .map(entity -> {
                    entity.deactivate();
                    QcInspectionPlan saved = planRepository.save(entity);
                    return ResponseEntity.ok(QcInspectionPlanDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/{id}/items")
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_UPDATE)
    public ResponseEntity<QcPlanItemDTO> addItem(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        
        return planRepository.findById(id)
                .map(plan -> {
                    Long itemId = ((Number) request.get("itemId")).longValue();
                    Integer itemSequence = (Integer) request.get("itemSequence");
                    
                    var planItem = com.metawebthree.mes.domain.entity.QcPlanItem.create(
                            id, itemId, itemSequence);
                    
                    if (request.containsKey("isMandatory")) {
                        planItem.setIsMandatory((Boolean) request.get("isMandatory"));
                    }
                    if (request.containsKey("defaultValue")) {
                        planItem.setDefaultValue((String) request.get("defaultValue"));
                    }
                    if (request.containsKey("inspectionMethod")) {
                        planItem.setInspectionMethod((String) request.get("inspectionMethod"));
                    }
                    if (request.containsKey("samplingRule")) {
                        planItem.setSamplingRule((String) request.get("samplingRule"));
                    }
                    
                    QcPlanItemDTO dto = QcPlanItemDTO.fromEntity(planItemRepository.save(planItem));
                    itemRepository.findById(itemId)
                            .ifPresent(inspectionItem -> 
                                dto.setInspectionItem(QcInspectionItemDTO.fromEntity(inspectionItem)));
                    return ResponseEntity.status(201).body(dto);
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @DeleteMapping("/{planId}/items/{itemId}")
    @RequirePermission(MesPermissions.QC_INSPECTION_PLAN_UPDATE)
    public ResponseEntity<Void> removeItem(
            @PathVariable Long planId,
            @PathVariable Long itemId) {
        
        List<com.metawebthree.mes.domain.entity.QcPlanItem> items = planItemRepository.findByPlanId(planId);
        items.stream()
                .filter(item -> item.getItemId().equals(itemId))
                .findFirst()
                .ifPresent(item -> planItemRepository.deleteById(item.getId()));
        
        return ResponseEntity.noContent().build();
    }
}