package com.metawebthree.rma.application.event;

public class RmaCompletedEvent {

    private final String eventType;
    private final Long rmaId;
    private final String rmaNo;

    public RmaCompletedEvent(String eventType, Long rmaId, String rmaNo) {
        this.eventType = eventType;
        this.rmaId = rmaId;
        this.rmaNo = rmaNo;
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
}
