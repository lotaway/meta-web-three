package com.metawebthree.rma.application.event;

import org.springframework.context.ApplicationEvent;

public class RmaCreatedEvent extends ApplicationEvent {

    private final Long rmaId;
    private final String rmaNo;

    public RmaCreatedEvent(Object source, Long rmaId, String rmaNo) {
        super(source);
        this.rmaId = rmaId;
        this.rmaNo = rmaNo;
    }

    public Long getRmaId() {
        return rmaId;
    }

    public String getRmaNo() {
        return rmaNo;
    }
}
