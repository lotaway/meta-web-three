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
public class EquipmentController {
    
    private final EquipmentRepository equipmentRepository;
    
    @PostMapping
    public ResponseEntity<EquipmentDTO> create(@RequestBody EquipmentDTO.CreateRequest request) {
        Equipment equipment = new Equipment();
        
        if (request.getEquipmentTypeId() != null) {
            equipment.create(request.getEquipmentCode(), request.getEquipmentName(),
                request.getEquipmentTypeId(), request.getEquipmentTypeCode(), request.getWorkshopId());
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
    
    @PutMapping("/{id}")
    public ResponseEntity<EquipmentDTO> update(
            @PathVariable Long id,
            @RequestBody EquipmentDTO.UpdateRequest request) {
        
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.setEquipmentName(request.getEquipmentName());
        equipment.setEquipmentTypeId(request.getEquipmentTypeId());
        equipment.setEquipmentTypeCode(request.getEquipmentTypeCode());
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
        
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
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
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
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
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
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
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
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
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
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
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        try {
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
        
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.bindDigitalTwinDevice(request.getDeviceCode());
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/unbind-digital-twin")
    public ResponseEntity<EquipmentDTO> unbindDigitalTwin(@PathVariable Long id) {
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.unbindDigitalTwinDevice();
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/heartbeat")
    public ResponseEntity<EquipmentDTO> heartbeat(@PathVariable Long id) {
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.heartbeat();
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/update-position")
    public ResponseEntity<EquipmentDTO> updatePosition(
            @PathVariable Long id,
            @RequestBody EquipmentDTO.UpdatePositionRequest request) {
        
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.updatePosition(request.getX(), request.getY(), request.getZ(), request.getRotation());
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
    
    @PostMapping("/{id}/calculate-oee")
    public ResponseEntity<EquipmentDTO> calculateOEE(
            @PathVariable Long id,
            @RequestBody EquipmentDTO.OEECalculationRequest request) {
        
        Equipment equipment = equipmentRepository.findById(id)
            .orElseThrow(() -> EquipmentException.notFound(id));
        
        equipment.calculateOEE(
            request.getPlannedProductionTime(),
            request.getIdealCycleTime(),
            request.getGoodProductCount()
        );
        equipmentRepository.update(equipment);
        
        return ResponseEntity.ok(EquipmentDTO.fromEntity(equipment));
    }
}