package com.metawebthree.mes.domain.event;

public class EquipmentBreakdownEvent extends MesEvent {
    private final Long equipmentId;

    public EquipmentBreakdownEvent(Object source, Long equipmentId) {
        super(source, MesEventType.EQUIPMENT_BREAKDOWN);
        this.equipmentId = equipmentId;
    }

    public Long getEquipmentId() {
        return equipmentId;
    }
}