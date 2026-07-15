package com.metawebthree.rma.infrastructure.event;

import com.metawebthree.rma.application.event.RmaCompletedEvent;
import com.metawebthree.rma.application.event.RmaCreatedEvent;
import com.metawebthree.rma.application.event.RmaDispositionExecutedEvent;
import com.metawebthree.rma.application.event.RmaInspectionCompletedEvent;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Component;

@Component
public class RmaDomainEventPublisher {

    private final ApplicationEventPublisher eventPublisher;

    public RmaDomainEventPublisher(ApplicationEventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publish(RmaCreatedEvent event) {
        eventPublisher.publishEvent(event);
    }

    public void publish(RmaInspectionCompletedEvent event) {
        eventPublisher.publishEvent(event);
    }

    public void publish(RmaDispositionExecutedEvent event) {
        eventPublisher.publishEvent(event);
    }

    public void publish(RmaCompletedEvent event) {
        eventPublisher.publishEvent(event);
    }
}
