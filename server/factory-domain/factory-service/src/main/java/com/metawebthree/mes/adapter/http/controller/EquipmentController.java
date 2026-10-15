package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.domain.entity.Equipment;
import com.metawebthree.mes.domain.exception.EquipmentException;
import com.metawebthree.mes.domain.repository.EquipmentRepository;
import com.metawebthree.mes.interfaces.dto.EquipmentDTO;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/equipment")
@RequiredArgsConstructor
@org.springframework.transaction.annotation.Transactional
public class EquipmentController {
    
    private final EquipmentRepository equipmentRepository;
    private final com.metawebthree.mes.domain.repository.EquipmentTypeRepository equipmentTypeRepository;
    private final com.metawebthree.mes.domain.service.EquipmentTransitionPolicy transitionPolicy;
    private final com.metawebthree.mes.domain.service.EquipmentStatusService statusService;
    
    @PostMapping
    public ResponseEntity<EquipmentDTO> create(@RequestBody EquipmentDTO.CreateRequest request) {
        Equipment equipment = new Equipment();
        
        if (request.getEquipmentTypeId() != null) {
            var type = equipmentTypeRepository.findById(request.getEquipmentTypeId())
                    .filter(t -> Boolean.TRUE.equals(t.getIsActive()))
                    .orElseThrow(() -> new IllegalArgumentException("Active equipment type required"));
            equipment.create(request.getEquipmentCode(), request.getEquipmentName(),
                type.getId(), type.getTypeCode(), request.getWorkshopId());
            String initial = transitionPolicy.initial(type.getId())
                    .orElseThrow(() -> new IllegalStateException("Equipment type requires an initial status"));
            equipment.transitionTo(initial, transitionPolicy.configId(type.getId(), initial));
        } else {
            equipment.create(request.getEquipmentCode(), request.getEquipmentName(),
                request.getEquipmentTypeCode(), request.getWorkshopId());
        }
        
        if (request.getWorkstationId() != null) {
            equipment.bindWorkstation(request.getWorkstationId());
        }
        
        Equipment saved = equipmentRepository.save(equipment);
        return ResponseEntity.ok(EquipmentDTO.fromEntity(saved));
    }
    
    public record TransitionRequest(@jakarta.validation.constraints.NotBlank String targetStatus) {}

    @PostMapping("/{id}/transition")
    public ResponseEntity<EquipmentDTO> transition(@PathVariable Long id,
            @jakarta.validation.Valid @RequestBody TransitionRequest request) {
        if (!statusService.transitionToStatus(id, request.targetStatus()))
            throw new IllegalStateException("Equipment transition is not permitted by active configuration");
        return getById(id);
    }

    @PutMapping("/{id}")
    public ResponseEntity<EquipmentDTO> update(
            @PathVariable Long id,
            @RequestBody EquipmentDTO.UpdateRequest request) {
        
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.setEquipmentName(request.getEquipmentName());
        if (!java.util.Objects.equals(equipment.getEquipmentTypeId(), request.getEquipmentTypeId()))
            throw new IllegalArgumentException("Equipment type cannot be changed through metadata update");
        equipment.setEquipmentTypeId(request.getEquipmentTypeId());
        if (equipment.getEquipmentTypeId() == null) equipment.setEquipmentTypeCode(request.getEquipmentTypeCode());
        equipment.setWorkshopId(request.getWorkshopId());
        
        if (request.getWorkstationId() != null) {
            equipment.bindWorkstation(request.getWorkstationId());
        } else {
            equipment.unbindWorkstation();
        }
        
        equipment.setIpAddress(request.getIpAddress());
        equipment.setMacAddress(request.getMacAddress());
        equipment.setMqttTopic(request.getMqttTopic());
        
        equipmentRepository.update(equipment);
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        equipmentRepository.deleteById(id);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<EquipmentDTO> getById(@PathVariable Long id) {
        return equipmentRepository.findById(id)
            .map(equipment -> ResponseEntity.ok(EquipmentDTO.fromEntity(equipment)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code/{equipmentCode}")
    public ResponseEntity<EquipmentDTO> getByCode(@PathVariable String equipmentCode) {
        return equipmentRepository.findByEquipmentCode(equipmentCode)
            .map(equipment -> ResponseEntity.ok(EquipmentDTO.fromEntity(equipment)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping
    public ResponseEntity<List<EquipmentDTO>> list(
            @RequestParam(required = false) String workshopId,
            @RequestParam(required = false) String statusCode,
            @RequestParam(required = false) String workstationId,
            @RequestParam(required = false) Long equipmentTypeId) {
        
        List<Equipment> equipments;
        
        if (workshopId != null && !workshopId.isEmpty()) {
            equipments = equipmentRepository.findByWorkshopId(workshopId);
        } else if (statusCode != null && !statusCode.isEmpty()) {
            equipments = equipmentRepository.findByStatusCode(statusCode);
        } else if (workstationId != null && !workstationId.isEmpty()) {
            equipments = equipmentRepository.findByWorkstationId(workstationId);
        } else if (equipmentTypeId != null) {
            equipments = equipmentRepository.findByEquipmentTypeId(equipmentTypeId);
        } else {
            return ResponseEntity.ok(List.of());
        }
        
        List<EquipmentDTO> dtos = equipments.stream()
            .map(EquipmentDTO::fromEntity)
            .collect(Collectors.toList());
        
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping("/{id}/start-task")
    public ResponseEntity<EquipmentDTO> startTask(
            @PathVariable Long id,
            @RequestBody EquipmentDTO.StartTaskRequest request) {
        
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
            requireConfiguredTransition(equipment, "RUNNING");
            equipment.startTask(request.getTaskNo());
        } catch (IllegalStateException e) {
            throw EquipmentException.invalidState(
                equipment.getStatus() != null ? equipment.getStatus().name() : "UNKNOWN",
                "startTask");
        }
        
        equipmentRepository.update(equipment);
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/complete-task")
    public ResponseEntity<EquipmentDTO> completeTask(@PathVariable Long id) {
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
            requireConfiguredTransition(equipment, "IDLE");
            equipment.completeTask();
        } catch (IllegalStateException e) {
            throw EquipmentException.invalidState(
                equipment.getStatus() != null ? equipment.getStatus().name() : "UNKNOWN",
                "completeTask");
        }
        
        equipmentRepository.update(equipment);
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/report-breakdown")
    public ResponseEntity<EquipmentDTO> reportBreakdown(@PathVariable Long id) {
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
            requireConfiguredTransition(equipment, "BREAKDOWN");
            equipment.reportBreakdown();
        } catch (IllegalStateException e) {
            throw EquipmentException.invalidState(
                equipment.getStatus() != null ? equipment.getStatus().name() : "UNKNOWN",
                "reportBreakdown");
        }
        
        equipmentRepository.update(equipment);
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/repair")
    public ResponseEntity<EquipmentDTO> repair(@PathVariable Long id) {
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
            requireConfiguredTransition(equipment, "IDLE");
            equipment.repair();
        } catch (IllegalStateException e) {
            throw EquipmentException.invalidState(
                equipment.getStatus() != null ? equipment.getStatus().name() : "UNKNOWN",
                "repair");
        }
        
        equipmentRepository.update(equipment);
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/start-maintenance")
    public ResponseEntity<EquipmentDTO> startMaintenance(@PathVariable Long id) {
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
            requireConfiguredTransition(equipment, "MAINTENANCE");
            equipment.startMaintenance();
        } catch (IllegalStateException e) {
            throw EquipmentException.invalidState(
                equipment.getStatus() != null ? equipment.getStatus().name() : "UNKNOWN",
                "startMaintenance");
        }
        
        equipmentRepository.update(equipment);
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/complete-maintenance")
    public ResponseEntity<EquipmentDTO> completeMaintenance(@PathVariable Long id) {
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
            requireConfiguredTransition(equipment, "IDLE");
            equipment.completeMaintenance();
        } catch (IllegalStateException e) {
            throw EquipmentException.invalidState(
                equipment.getStatus() != null ? equipment.getStatus().name() : "UNKNOWN",
                "completeMaintenance");
        }
        
        equipmentRepository.update(equipment);
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/bind-digital-twin")
    public ResponseEntity<EquipmentDTO> bindDigitalTwin(
            @PathVariable Long id,
            @RequestBody EquipmentDTO.BindDigitalTwinRequest request) {
        
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.bindDigitalTwinDevice(request.getDeviceCode());
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/unbind-digital-twin")
    public ResponseEntity<EquipmentDTO> unbindDigitalTwin(@PathVariable Long id) {
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.unbindDigitalTwinDevice();
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/heartbeat")
    public ResponseEntity<EquipmentDTO> heartbeat(@PathVariable Long id) {
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        if ("OFFLINE".equals(equipment.getStatusCode())) requireConfiguredTransition(equipment, "ONLINE");
        equipment.heartbeat();
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/update-position")
    public ResponseEntity<EquipmentDTO> updatePosition(
            @PathVariable Long id,
            @RequestBody EquipmentDTO.UpdatePositionRequest request) {
        
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.updatePosition(request.getX(), request.getY(), request.getZ(), request.getRotation());
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/calculate-oee")
    public ResponseEntity<EquipmentDTO> calculateOEE(
            @PathVariable Long id,
            @RequestBody EquipmentDTO.OEECalculationRequest request) {
        
        Equipment equipment = equipmentRepository.findByIdForUpdate(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.calculateOEE(
            request.getPlannedProductionTime(),
            request.getIdealCycleTime(),
            request.getGoodProductCount()
        );
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    private void requireConfiguredTransition(Equipment equipment, String target) {
        if (equipment.getEquipmentTypeId() != null && !transitionPolicy.allows(equipment, target))
            throw new IllegalStateException("Equipment action is not permitted by active configuration");
        if (equipment.getEquipmentTypeId() != null)
            equipment.setStatusConfigId(transitionPolicy.configId(equipment.getEquipmentTypeId(), target));
    }
}