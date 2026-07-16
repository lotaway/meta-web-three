package com.metawebthree.dom.domain.service;

import com.metawebthree.dom.domain.entity.DomOrder;
import com.metawebthree.dom.domain.entity.DomOrderLine;
import com.metawebthree.dom.domain.entity.FulfillmentPlan;
import java.util.List;

public interface DomDomainService {

    DomOrder createDomOrder(DomOrder order, List<DomOrderLine> lines);

    DomOrder checkAvailability(DomOrder order, List<DomOrderLine> lines);

    List<DomOrderLine> sourceOrder(DomOrder order, List<DomOrderLine> lines, String strategy);

    FulfillmentPlan createFulfillmentPlan(DomOrder order, List<DomOrderLine> sourcedLines);

    FulfillmentPlan approveFulfillmentPlan(Long planId);

    DomOrder cancelDomOrder(Long orderId);
}
