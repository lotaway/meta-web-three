package com.metawebthree.mes.domain.event;

public class EquipmentRepairedEvent extends MesEvent {
    private final Long equipmentId;

    public EquipmentRepairedEvent(Object source, Long equipmentId) {
        super(source, MesEventType.EQUIPMENT_REPAIRED);
        this.equipmentId = equipmentId;
    }

    public Long getEquipmentId() {
        return equipmentId;
    }
}