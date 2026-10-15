package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.config.*;
import com.metawebthree.mes.domain.entity.*;
import com.metawebthree.mes.domain.repository.*;
import org.springframework.stereotype.Component;
import java.util.Optional;
import java.util.Objects;

@Component
public class EquipmentTransitionPolicy {
    private final EquipmentTypeRepository types;
    private final EquipmentStatusConfigRepository statuses;
    private final EquipmentStatusTransitionRepository transitions;
    private final StatusMachineRepository machines;
    public EquipmentTransitionPolicy(EquipmentTypeRepository types, EquipmentStatusConfigRepository statuses,
            EquipmentStatusTransitionRepository transitions, StatusMachineRepository machines) {
        this.types=types; this.statuses=statuses; this.transitions=transitions; this.machines=machines;
    }
    public boolean allows(Equipment equipment,String target) {
        if(equipment==null || equipment.getEquipmentTypeId()==null || target==null || target.isBlank()
                || Objects.equals(target,equipment.getStatusCode())) return false;
        EquipmentType type=types.findById(equipment.getEquipmentTypeId()).orElse(null);
        if(type==null || !Boolean.TRUE.equals(type.getIsActive())) return false;
        if(type.getStatusMachineId()!=null) {
            StatusMachine machine=machines.findById(type.getStatusMachineId()).orElse(null);
            if(machine==null || !"ACTIVE".equals(machine.getStatus()) || !"EQUIPMENT".equals(machine.getEntityType())
                    || machine.getStatuses()==null || machine.getTransitions()==null) return false;
            StatusConfig source=machine.getStatuses().stream().filter(s -> Objects.equals(s.getStatusCode(),equipment.getStatusCode())).findFirst().orElse(null);
            boolean targetExists=machine.getStatuses().stream().anyMatch(s -> Objects.equals(s.getStatusCode(),target));
            if(source==null || Boolean.TRUE.equals(source.getIsFinal()) || !targetExists) return false;
            return machine.getTransitions().stream().anyMatch(t -> Objects.equals(t.getFromStatus(),equipment.getStatusCode())
                    && Objects.equals(t.getToStatus(),target) && empty(t.getConditionExpression()) && empty(t.getEventCode())
                    && !Boolean.TRUE.equals(t.getIsAutoTransition()) && passive(t.getTransitionAction()));
        }
        EquipmentStatusConfig source=statuses.findByEquipmentTypeIdAndStatusCode(type.getId(),equipment.getStatusCode()).orElse(null);
        EquipmentStatusConfig destination=statuses.findByEquipmentTypeIdAndStatusCode(type.getId(),target).orElse(null);
        if(source==null || destination==null || !Boolean.TRUE.equals(source.getIsActive())
                || !Boolean.TRUE.equals(destination.getIsActive()) || Boolean.TRUE.equals(source.getIsFinal())) return false;
        return transitions.findByEquipmentTypeIdAndFromStatusCodeAndToStatusCode(type.getId(),equipment.getStatusCode(),target)
                .filter(t -> Boolean.TRUE.equals(t.getIsActive()) && empty(t.getConditionExpression()) && empty(t.getEventCode())
                        && !Boolean.TRUE.equals(t.getIsAutoTransition()) && t.getTimeoutSeconds()==null
                        && empty(t.getTimeoutAction()) && passive(t.getTransitionAction())).isPresent();
    }
    public Optional<String> initial(Long typeId) {
        EquipmentType type=types.findById(typeId).filter(t -> Boolean.TRUE.equals(t.getIsActive())).orElseThrow(() -> new IllegalArgumentException("Active equipment type required"));
        if(type.getStatusMachineId()!=null) {
            StatusMachine machine=machines.findById(type.getStatusMachineId()).orElseThrow(() -> new IllegalArgumentException("Bound machine not found"));
            if(!"ACTIVE".equals(machine.getStatus()) || !"EQUIPMENT".equals(machine.getEntityType())) throw new IllegalStateException("Bound machine must be active EQUIPMENT machine");
            return machine.getStatuses().stream().filter(s -> Boolean.TRUE.equals(s.getIsInitial())).map(StatusConfig::getStatusCode).findFirst();
        }
        return statuses.findByEquipmentTypeIdAndIsInitial(typeId,true).filter(s -> Boolean.TRUE.equals(s.getIsActive())).map(EquipmentStatusConfig::getStatusCode);
    }
    public Long configId(Long typeId,String code) {
        EquipmentType type=types.findById(typeId).orElseThrow();
        // Generic machine IDs belong to a different configuration namespace.
        if(type.getStatusMachineId()!=null) return null;
        return statuses.findByEquipmentTypeIdAndStatusCode(typeId,code).map(EquipmentStatusConfig::getId).orElse(null);
    }
    private static boolean empty(String value) { return value==null || value.isBlank(); }
    private static boolean passive(String value) { return empty(value) || "MANUAL".equals(value) || "SET_STATUS".equals(value); }
}
