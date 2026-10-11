package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.domain.entity.Workstation;
import com.metawebthree.mes.domain.repository.WorkstationRepository;
import com.metawebthree.mes.interfaces.dto.WorkstationDTO;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/workstation")
@RequiredArgsConstructor
public class WorkstationController {
    
    private final WorkstationRepository workstationRepository;
    
    @PostMapping
    public ResponseEntity<WorkstationDTO> create(@RequestBody WorkstationDTO.CreateRequest request) {
        Workstation workstation = new Workstation();
        workstation.create(
            request.getWorkstationCode(),
            request.getWorkstationName(),
            request.getWorkshopId(),
            Workstation.WorkstationType.valueOf(request.getType())
        );
        workstation.setWorkshopName(request.getWorkshopName());
        workstation.setLocation(request.getLocation());
        workstation.setCapacity(request.getCapacity());
        workstation.setDescription(request.getDescription());
        
        Workstation saved = workstationRepository.save(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<WorkstationDTO> update(
            @PathVariable Long id,
            @RequestBody WorkstationDTO.UpdateRequest request) {
        
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        if (request.getWorkstationName() != null) {
            workstation.setWorkstationName(request.getWorkstationName());
        }
        if (request.getWorkshopId() != null) {
            workstation.setWorkshopId(request.getWorkshopId());
        }
        if (request.getWorkshopName() != null) {
            workstation.setWorkshopName(request.getWorkshopName());
        }
        if (request.getType() != null) {
            workstation.setType(Workstation.WorkstationType.valueOf(request.getType()));
        }
        if (request.getStatus() != null) {
            workstation.setStatus(Workstation.WorkstationStatus.valueOf(request.getStatus()));
        }
        if (request.getLocation() != null) {
            workstation.setLocation(request.getLocation());
        }
        if (request.getCapacity() != null) {
            workstation.setCapacity(request.getCapacity());
        }
        if (request.getDescription() != null) {
            workstation.setDescription(request.getDescription());
        }
        
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        workstationRepository.deleteById(id);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<WorkstationDTO> getById(@PathVariable Long id) {
        return workstationRepository.findById(id)
            .map(workstation -> ResponseEntity.ok(WorkstationDTO.fromEntity(workstation)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code/{workstationCode}")
    public ResponseEntity<WorkstationDTO> getByCode(@PathVariable String workstationCode) {
        return workstationRepository.findByWorkstationCode(workstationCode)
            .map(workstation -> ResponseEntity.ok(WorkstationDTO.fromEntity(workstation)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/workshop/{workshopId}")
    public ResponseEntity<List<WorkstationDTO>> getByWorkshop(@PathVariable String workshopId) {
        List<WorkstationDTO> workstations = workstationRepository.findByWorkshopId(workshopId)
            .stream()
            .map(WorkstationDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(workstations);
    }
    
    @GetMapping("/status/{status}")
    public ResponseEntity<List<WorkstationDTO>> getByStatus(@PathVariable String status) {
        List<WorkstationDTO> workstations = workstationRepository.findByStatus(Workstation.WorkstationStatus.valueOf(status))
            .stream()
            .map(WorkstationDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(workstations);
    }
    
    @GetMapping("/type/{type}")
    public ResponseEntity<List<WorkstationDTO>> getByType(@PathVariable String type) {
        List<WorkstationDTO> workstations = workstationRepository.findByType(Workstation.WorkstationType.valueOf(type))
            .stream()
            .map(WorkstationDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(workstations);
    }
    
    @GetMapping
    public ResponseEntity<List<WorkstationDTO>> getAll() {
        List<WorkstationDTO> workstations = workstationRepository.findByStatus(Workstation.WorkstationStatus.ACTIVE)
            .stream()
            .map(WorkstationDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(workstations);
    }
    
    @PostMapping("/{id}/activate")
    public ResponseEntity<WorkstationDTO> activate(@PathVariable Long id) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.activate();
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @PostMapping("/{id}/deactivate")
    public ResponseEntity<WorkstationDTO> deactivate(@PathVariable Long id) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.deactivate();
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @PostMapping("/{id}/maintenance")
    public ResponseEntity<WorkstationDTO> setMaintenance(@PathVariable Long id) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.setMaintenance();
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @PostMapping("/{id}/fault")
    public ResponseEntity<WorkstationDTO> setFault(@PathVariable Long id) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.setFault();
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @PostMapping("/{id}/bind-equipment")
    public ResponseEntity<WorkstationDTO> bindEquipment(
            @PathVariable Long id,
            @RequestBody WorkstationDTO.BindEquipmentRequest request) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.bindEquipment(request.getEquipmentId(), request.getEquipmentCode());
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @PostMapping("/{id}/unbind-equipment")
    public ResponseEntity<WorkstationDTO> unbindEquipment(
            @PathVariable Long id,
            @RequestBody WorkstationDTO.BindEquipmentRequest request) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.unbindEquipment(request.getEquipmentId());
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @PostMapping("/{id}/bind-tool")
    public ResponseEntity<WorkstationDTO> bindTool(
            @PathVariable Long id,
            @RequestBody WorkstationDTO.BindToolRequest request) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.bindTool(request.getToolId(), request.getToolName());
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @PostMapping("/{id}/unbind-tool")
    public ResponseEntity<WorkstationDTO> unbindTool(
            @PathVariable Long id,
            @RequestBody WorkstationDTO.BindToolRequest request) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.unbindTool(request.getToolId());
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @PostMapping("/{id}/bind-operator")
    public ResponseEntity<WorkstationDTO> bindOperator(
            @PathVariable Long id,
            @RequestBody WorkstationDTO.BindOperatorRequest request) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.bindOperator(request.getOperatorId(), request.getOperatorName());
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
    
    @PostMapping("/{id}/unbind-operator")
    public ResponseEntity<WorkstationDTO> unbindOperator(
            @PathVariable Long id,
            @RequestBody WorkstationDTO.BindOperatorRequest request) {
        Workstation workstation = workstationRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Workstation not found: " + id));
        
        workstation.unbindOperator(request.getOperatorId());
        workstationRepository.update(workstation);
        return ResponseEntity.ok(WorkstationDTO.fromEntity(workstation));
    }
}