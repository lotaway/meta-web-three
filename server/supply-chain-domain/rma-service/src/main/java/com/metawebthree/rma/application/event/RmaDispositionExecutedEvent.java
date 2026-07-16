package com.metawebthree.rma.application.event;

import org.springframework.context.ApplicationEvent;

public class RmaDispositionExecutedEvent extends ApplicationEvent {

    private final Long rmaId;
    private final String rmaNo;
    private final String dispositionType;

    public RmaDispositionExecutedEvent(Object source, Long rmaId, String rmaNo, String dispositionType) {
        super(source);
        this.rmaId = rmaId;
        this.rmaNo = rmaNo;
        this.dispositionType = dispositionType;
    }

    public Long getRmaId() {
        return rmaId;
    }

    public String getRmaNo() {
        return rmaNo;
    }

    public String getDispositionType() {
        return dispositionType;
    }
}
