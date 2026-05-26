package com.metawebthree.mes.domain.event;

import org.springframework.context.ApplicationEvent;

public abstract class MesEvent extends ApplicationEvent {
    private final MesEventType eventType;

    public MesEvent(Object source, MesEventType eventType) {
        super(source);
        this.eventType = eventType;
    }

    public MesEventType getEventType() {
        return eventType;
    }
}