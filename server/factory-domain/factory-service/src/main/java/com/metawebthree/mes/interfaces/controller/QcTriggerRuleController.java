package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.entity.QcTriggerRule;
import com.metawebthree.mes.domain.entity.QcTriggerRule.TriggerType;
import com.metawebthree.mes.domain.repository.QcTriggerRuleRepository;
import com.metawebthree.mes.interfaces.dto.QcTriggerRuleDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/qc/trigger-rule")
public class QcTriggerRuleController {
    
    private final QcTriggerRuleRepository qcTriggerRuleRepository;
    
    public QcTriggerRuleController(QcTriggerRuleRepository qcTriggerRuleRepository) {
        this.qcTriggerRuleRepository = qcTriggerRuleRepository;
    }
    
    @GetMapping
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_READ)
    public ResponseEntity<List<QcTriggerRuleDTO>> getAll() {
        List<QcTriggerRuleDTO> dtos = qcTriggerRuleRepository.findAll().stream()
                .map(QcTriggerRuleDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/{id}")
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_READ)
    public ResponseEntity<QcTriggerRuleDTO> getById(@PathVariable Long id) {
        return qcTriggerRuleRepository.findById(id)
                .map(entity -> ResponseEntity.ok(QcTriggerRuleDTO.fromEntity(entity)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code/{code}")
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_READ)
    public ResponseEntity<QcTriggerRuleDTO> getByCode(@PathVariable String code) {
        return qcTriggerRuleRepository.findByRuleCode(code)
                .map(entity -> ResponseEntity.ok(QcTriggerRuleDTO.fromEntity(entity)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/type/{type}")
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_READ)
    public ResponseEntity<List<QcTriggerRuleDTO>> getByTriggerType(@PathVariable String type) {
        TriggerType triggerType = TriggerType.valueOf(type);
        List<QcTriggerRuleDTO> dtos = qcTriggerRuleRepository.findByTriggerType(triggerType).stream()
                .map(QcTriggerRuleDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/enabled")
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_READ)
    public ResponseEntity<List<QcTriggerRuleDTO>> getEnabled() {
        List<QcTriggerRuleDTO> dtos = qcTriggerRuleRepository.findByIsEnabled(true).stream()
                .map(QcTriggerRuleDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_CREATE)
    public ResponseEntity<QcTriggerRuleDTO> create(@RequestBody Map<String, Object> request) {
        String ruleCode = (String) request.get("ruleCode");
        String ruleName = (String) request.get("ruleName");
        String triggerTypeStr = (String) request.get("triggerType");
        String targetObject = (String) request.get("targetObject");
        
        if (qcTriggerRuleRepository.existsByRuleCode(ruleCode)) {
            return ResponseEntity.badRequest().build();
        }
        
        QcTriggerRule entity = new QcTriggerRule();
        entity.create(ruleCode, ruleName, TriggerType.valueOf(triggerTypeStr), targetObject);
        
        if (request.containsKey("inspectionType")) {
            entity.setInspectionType((String) request.get("inspectionType"));
        }
        if (request.containsKey("inspectionPlanCode")) {
            entity.setInspectionPlanCode((String) request.get("inspectionPlanCode"));
        }
        if (request.containsKey("priority")) {
            entity.setPriority((Integer) request.get("priority"));
        }
        
        if (request.containsKey("condition")) {
            @SuppressWarnings("unchecked")
            Map<String, Object> condMap = (Map<String, Object>) request.get("condition");
            QcTriggerRuleDTO.TriggerConditionDTO cond = new QcTriggerRuleDTO.TriggerConditionDTO();
            if (condMap.containsKey("batchSize")) {
                entity.setBatchTrigger((Integer) condMap.get("batchSize"));
            } else if (condMap.containsKey("timeIntervalMinutes")) {
                entity.setTimeTrigger((Integer) condMap.get("timeIntervalMinutes"));
            } else if (condMap.containsKey("quantityThreshold")) {
                entity.setQuantityTrigger((Integer) condMap.get("quantityThreshold"));
            } else if (condMap.containsKey("eventType")) {
                entity.setEventTrigger((String) condMap.get("eventType"));
            } else if (condMap.containsKey("cronExpression")) {
                entity.setCronTrigger((String) condMap.get("cronExpression"));
            }
        }
        
        QcTriggerRule saved = qcTriggerRuleRepository.save(entity);
        return ResponseEntity.status(201).body(QcTriggerRuleDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_UPDATE)
    public ResponseEntity<QcTriggerRuleDTO> update(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        
        return qcTriggerRuleRepository.findById(id)
                .map(entity -> {
                    if (request.containsKey("ruleName")) {
                        entity.setRuleName((String) request.get("ruleName"));
                    }
                    if (request.containsKey("targetObject")) {
                        entity.setTargetObject((String) request.get("targetObject"));
                    }
                    if (request.containsKey("inspectionType")) {
                        entity.setInspectionType((String) request.get("inspectionType"));
                    }
                    if (request.containsKey("inspectionPlanCode")) {
                        entity.setInspectionPlanCode((String) request.get("inspectionPlanCode"));
                    }
                    if (request.containsKey("priority")) {
                        entity.updatePriority((Integer) request.get("priority"));
                    }
                    
                    if (request.containsKey("condition")) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> condMap = (Map<String, Object>) request.get("condition");
                        if (condMap.containsKey("batchSize")) {
                            entity.setBatchTrigger((Integer) condMap.get("batchSize"));
                        } else if (condMap.containsKey("timeIntervalMinutes")) {
                            entity.setTimeTrigger((Integer) condMap.get("timeIntervalMinutes"));
                        } else if (condMap.containsKey("quantityThreshold")) {
                            entity.setQuantityTrigger((Integer) condMap.get("quantityThreshold"));
                        } else if (condMap.containsKey("eventType")) {
                            entity.setEventTrigger((String) condMap.get("eventType"));
                        } else if (condMap.containsKey("cronExpression")) {
                            entity.setCronTrigger((String) condMap.get("cronExpression"));
                        }
                    }
                    
                    QcTriggerRule saved = qcTriggerRuleRepository.save(entity);
                    return ResponseEntity.ok(QcTriggerRuleDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @DeleteMapping("/{id}")
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_DELETE)
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        if (qcTriggerRuleRepository.findById(id).isPresent()) {
            qcTriggerRuleRepository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
    
    @PostMapping("/{id}/enable")
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_UPDATE)
    public ResponseEntity<QcTriggerRuleDTO> enable(@PathVariable Long id) {
        return qcTriggerRuleRepository.findById(id)
                .map(entity -> {
                    entity.enable();
                    QcTriggerRule saved = qcTriggerRuleRepository.save(entity);
                    return ResponseEntity.ok(QcTriggerRuleDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/{id}/disable")
    @RequirePermission(MesPermissions.QC_TRIGGER_RULE_UPDATE)
    public ResponseEntity<QcTriggerRuleDTO> disable(@PathVariable Long id) {
        return qcTriggerRuleRepository.findById(id)
                .map(entity -> {
                    entity.disable();
                    QcTriggerRule saved = qcTriggerRuleRepository.save(entity);
                    return ResponseEntity.ok(QcTriggerRuleDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
}