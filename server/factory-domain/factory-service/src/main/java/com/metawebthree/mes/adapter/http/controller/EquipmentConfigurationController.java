package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.domain.entity.*;
import com.metawebthree.mes.domain.config.StatusMachine;
import com.metawebthree.mes.domain.repository.*;
import com.metawebthree.mes.domain.service.*;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
import java.util.List;

@RestController
@RequestMapping("/api/mes/equipment-types")
@Transactional
public class EquipmentConfigurationController {
    private final EquipmentTypeService service;
    private final EquipmentTypeRepository types;
    private final EquipmentStatusConfigRepository statuses;
    private final StatusMachineRepository machines;
    public EquipmentConfigurationController(EquipmentTypeService service,EquipmentTypeRepository types,
            EquipmentStatusConfigRepository statuses,StatusMachineRepository machines) {
        this.service=service; this.types=types; this.statuses=statuses; this.machines=machines;
    }
    public record TypeRequest(@NotBlank String typeCode,@NotBlank String typeName,@NotBlank String category) {}
    public record StatusRequest(@NotBlank String statusCode,@NotBlank String statusName,@NotBlank String statusCategory,
            boolean initial,boolean terminal,boolean alarm) {}
    public record TransitionRequest(@NotBlank String fromStatus,@NotBlank String toStatus,
            @Pattern(regexp="MANUAL|SET_STATUS") String action) {}
    public record BindingRequest(@NotNull @Positive Long machineId) {}
    public record ActivationRequest(boolean active) {}
    @PostMapping public EquipmentType create(@Valid @RequestBody TypeRequest r) { return service.createType(r.typeCode(),r.typeName(),r.category()); }
    @GetMapping public List<EquipmentType> list() { return service.getAllTypes(); }
    @GetMapping("/{id}") public EquipmentType type(@PathVariable Long id) { return types.findById(id).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND)); }
    @PutMapping("/{id}/active") public EquipmentType activate(@PathVariable Long id,@RequestBody ActivationRequest r) {
        EquipmentType type=type(id); if(r.active()) type.activate(); else type.deactivate(); types.update(type); return type;
    }
    @PostMapping("/{id}/statuses") public EquipmentStatusConfig createStatus(@PathVariable Long id,@Valid @RequestBody StatusRequest r) {
        EquipmentStatusConfig config=service.addStatusConfig(id,r.statusCode(),r.statusName(),r.statusCategory());
        config.setIsInitial(r.initial()); config.setIsFinal(r.terminal()); config.setIsAlarm(r.alarm()); statuses.update(config); return config;
    }
    @GetMapping("/{id}/statuses") public List<EquipmentStatusConfig> listStatuses(@PathVariable Long id) { type(id); return service.getStatusConfigs(id); }
    @PutMapping("/{id}/statuses/{statusId}/active") public EquipmentStatusConfig activateStatus(@PathVariable Long id,@PathVariable Long statusId,@RequestBody ActivationRequest r) {
        EquipmentStatusConfig config=statuses.findById(statusId).filter(s -> id.equals(s.getEquipmentTypeId())).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND));
        config.setIsActive(r.active()); statuses.update(config); return config;
    }
    @PostMapping("/{id}/transitions") public EquipmentStatusTransition createTransition(@PathVariable Long id,@Valid @RequestBody TransitionRequest r) {
        return service.addStatusTransition(id,r.fromStatus(),r.toStatus(),r.action()==null?"MANUAL":r.action());
    }
    @GetMapping("/{id}/transitions") public List<EquipmentStatusTransition> transitions(@PathVariable Long id) { type(id); return service.getStatusTransitions(id); }
    @PutMapping("/{id}/status-machine") public EquipmentType bind(@PathVariable Long id,@Valid @RequestBody BindingRequest r) {
        EquipmentType type=type(id);
        StatusMachine machine=machines.findById(r.machineId()).orElseThrow(() -> new IllegalArgumentException("Machine not found"));
        if(!"ACTIVE".equals(machine.getStatus()) || !"EQUIPMENT".equals(machine.getEntityType())) throw new IllegalArgumentException("Active EQUIPMENT machine required");
        if(machine.getStatuses()==null || machine.getStatuses().stream().filter(s -> Boolean.TRUE.equals(s.getIsInitial())).count()!=1)
            throw new IllegalArgumentException("Machine must define exactly one initial status");
        type.bindStatusMachine(machine.getId()); types.update(type); return type;
    }
}
