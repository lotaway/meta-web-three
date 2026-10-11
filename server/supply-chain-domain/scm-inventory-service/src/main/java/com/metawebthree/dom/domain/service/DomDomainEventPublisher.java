package com.metawebthree.dom.domain.service;

import com.metawebthree.dom.domain.entity.DomOrder;
import com.metawebthree.dom.domain.entity.FulfillmentPlan;

public interface DomDomainEventPublisher {

    void publishDomOrderCreated(DomOrder order);

    void publishDomOrderSourced(DomOrder order, FulfillmentPlan plan);

    void publishDomOrderFulfilled(DomOrder order, FulfillmentPlan plan);
}
