package com.metawebthree.rma.application.event;

public class RmaInspectionCompletedEvent {

    private final String eventType;
    private final Long rmaId;
    private final String rmaNo;
    private final String inspectionResult;
    private final Integer acceptedQuantity;

    public RmaInspectionCompletedEvent(String eventType, Long rmaId, String rmaNo,
                                       String inspectionResult, Integer acceptedQuantity) {
        this.eventType = eventType;
        this.rmaId = rmaId;
        this.rmaNo = rmaNo;
        this.inspectionResult = inspectionResult;
        this.acceptedQuantity = acceptedQuantity;
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

    public String getInspectionResult() {
        return inspectionResult;
    }

    public Integer getAcceptedQuantity() {
        return acceptedQuantity;
    }
}
