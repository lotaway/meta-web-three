package com.metawebthree.rma.application.event;

import org.springframework.context.ApplicationEvent;

public class RmaInspectionCompletedEvent extends ApplicationEvent {

    private final Long rmaId;
    private final String rmaNo;
    private final String inspectionResult;
    private final Integer acceptedQuantity;

    public RmaInspectionCompletedEvent(Object source, Long rmaId, String rmaNo,
                                       String inspectionResult, Integer acceptedQuantity) {
        super(source);
        this.rmaId = rmaId;
        this.rmaNo = rmaNo;
        this.inspectionResult = inspectionResult;
        this.acceptedQuantity = acceptedQuantity;
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
