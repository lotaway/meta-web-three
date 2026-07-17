package com.metawebthree.mes.domain.event;

public class EquipmentBreakdownEvent extends MesEvent {
    private final Long equipmentId;

    public EquipmentBreakdownEvent(Long equipmentId) {
        super(MesEventType.EQUIPMENT_BREAKDOWN);
        this.equipmentId = equipmentId;
    }

    public Long getEquipmentId() {
        return equipmentId;
    }
}