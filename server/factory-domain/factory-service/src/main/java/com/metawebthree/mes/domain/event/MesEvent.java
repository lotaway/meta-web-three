package com.metawebthree.mes.domain.event;

public abstract class MesEvent {
    private final MesEventType eventType;

    public MesEvent(MesEventType eventType) {
        this.eventType = eventType;
    }

    public MesEventType getEventType() {
        return eventType;
    }
}