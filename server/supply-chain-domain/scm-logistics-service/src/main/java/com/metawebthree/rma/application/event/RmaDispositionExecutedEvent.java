package com.metawebthree.rma.application.event;

public class RmaDispositionExecutedEvent {

    private final String eventType;
    private final Long rmaId;
    private final String rmaNo;
    private final String dispositionType;

    public RmaDispositionExecutedEvent(String eventType, Long rmaId, String rmaNo, String dispositionType) {
        this.eventType = eventType;
        this.rmaId = rmaId;
        this.rmaNo = rmaNo;
        this.dispositionType = dispositionType;
    }

    public String getEventType() {
        return eventType;
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
