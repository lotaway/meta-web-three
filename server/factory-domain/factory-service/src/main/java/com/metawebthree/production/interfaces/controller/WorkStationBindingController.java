package com.metawebthree.production.interfaces.controller;

import com.metawebthree.production.application.WorkStationBindingService;
import com.metawebthree.production.domain.entity.WorkStationBinding;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/production/station-bindings")
public class WorkStationBindingController {
    
    private final WorkStationBindingService service;
    
    public WorkStationBindingController(WorkStationBindingService service) {
        this.service = service;
    }
    
    @PostMapping("/equipment")
    public ResponseEntity<WorkStationBinding> bindEquipment(@RequestBody Map<String, Object> request) {
        String workstationCode = (String) request.get("workstationCode");
        String equipmentCode = (String) request.get("equipmentCode");
        String equipmentName = (String) request.get("equipmentName");
        String equipmentType = (String) request.get("equipmentType");
        
        return ResponseEntity.ok(service.bindEquipment(
            workstationCode, equipmentCode, equipmentName, equipmentType));
    }
    
    @PostMapping("/tool")
    public ResponseEntity<WorkStationBinding> bindTool(@RequestBody Map<String, Object> request) {
        String workstationCode = (String) request.get("workstationCode");
        String toolCode = (String) request.get("toolCode");
        String toolName = (String) request.get("toolName");
        String toolType = (String) request.get("toolType");
        
        return ResponseEntity.ok(service.bindTool(
            workstationCode, toolCode, toolName, toolType));
    }
    
    @PostMapping("/personnel")
    public ResponseEntity<WorkStationBinding> bindPersonnel(@RequestBody Map<String, Object> request) {
        String workstationCode = (String) request.get("workstationCode");
        String personnelCode = (String) request.get("personnelCode");
        String personnelName = (String) request.get("personnelName");
        String personnelType = (String) request.get("personnelType");
        
        return ResponseEntity.ok(service.bindPersonnel(
            workstationCode, personnelCode, personnelName, personnelType));
    }
    
    @PostMapping("/{id}/primary")
    public ResponseEntity<Void> setPrimary(@PathVariable Long id) {
        service.setPrimaryBinding(id);
        return ResponseEntity.ok().build();
    }
    
    @PostMapping("/{id}/unbind")
    public ResponseEntity<Void> unbind(@PathVariable Long id) {
        service.unbind(id);
        return ResponseEntity.ok().build();
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteBinding(@PathVariable Long id) {
        service.deleteBinding(id);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/workstation/{workstationCode}")
    public ResponseEntity<List<WorkStationBinding>> getBindingsByWorkstation(
            @PathVariable String workstationCode) {
        return ResponseEntity.ok(service.getBindingsByWorkstation(workstationCode));
    }
    
    @GetMapping("/workstation/{workstationCode}/equipment")
    public ResponseEntity<List<WorkStationBinding>> getEquipmentBindings(
            @PathVariable String workstationCode) {
        return ResponseEntity.ok(service.getEquipmentBindings(workstationCode));
    }
    
    @GetMapping("/workstation/{workstationCode}/tool")
    public ResponseEntity<List<WorkStationBinding>> getToolBindings(
            @PathVariable String workstationCode) {
        return ResponseEntity.ok(service.getToolBindings(workstationCode));
    }
    
    @GetMapping("/workstation/{workstationCode}/personnel")
    public ResponseEntity<List<WorkStationBinding>> getPersonnelBindings(
            @PathVariable String workstationCode) {
        return ResponseEntity.ok(service.getPersonnelBindings(workstationCode));
    }
    
    @GetMapping("/workstation/{workstationCode}/equipment/primary")
    public ResponseEntity<WorkStationBinding> getPrimaryEquipment(
            @PathVariable String workstationCode) {
        return ResponseEntity.ok(service.getPrimaryEquipment(workstationCode));
    }
}