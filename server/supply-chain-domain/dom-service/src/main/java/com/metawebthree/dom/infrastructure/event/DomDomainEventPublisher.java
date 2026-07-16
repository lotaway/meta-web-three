package com.metawebthree.dom.infrastructure.event;

import com.metawebthree.dom.domain.entity.DomOrder;
import com.metawebthree.dom.domain.entity.FulfillmentPlan;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class DomDomainEventPublisher {

    private static final Logger log = LoggerFactory.getLogger(DomDomainEventPublisher.class);

    public void publishDomOrderCreated(DomOrder order) {
        log.info("[DOM EVENT] Order created: domOrderNo={}, customerId={}, totalAmount={}",
                order.getDomOrderNo(), order.getCustomerId(), order.getTotalAmount());
    }

    public void publishDomOrderSourced(DomOrder order, FulfillmentPlan plan) {
        log.info("[DOM EVENT] Order sourced: domOrderNo={}, planId={}, fulfillmentLines={}",
                order.getDomOrderNo(), plan.getId(), plan.getFulfilledLines());
    }

    public void publishDomOrderFulfilled(DomOrder order, FulfillmentPlan plan) {
        log.info("[DOM EVENT] Order fulfilled: domOrderNo={}, planId={}",
                order.getDomOrderNo(), plan.getId());
    }
}
