package com.metawebthree.dom.domain.service;

import com.metawebthree.dom.domain.entity.DomOrder;
import com.metawebthree.dom.domain.entity.DomOrderLine;
import com.metawebthree.dom.domain.entity.FulfillmentPlan;
import com.metawebthree.dom.domain.entity.SourcingStrategy;
import java.util.List;

public interface DomDomainService {

    DomOrder createDomOrder(DomOrder order, List<DomOrderLine> lines);

    void saveDomOrder(DomOrder order, List<DomOrderLine> lines);

    boolean checkAvailability(DomOrder order, List<DomOrderLine> lines);

    void saveAvailabilityResult(DomOrder order, List<DomOrderLine> lines, boolean allPass);

    List<DomOrderLine> sourceOrder(DomOrder order, List<DomOrderLine> lines, SourcingStrategy strategy);

    void saveSourcingResult(DomOrder order, List<DomOrderLine> lines);

    FulfillmentPlan createFulfillmentPlan(DomOrder order, List<DomOrderLine> sourcedLines);

    void saveFulfillmentPlan(DomOrder order, FulfillmentPlan plan);

    FulfillmentPlan approveFulfillmentPlan(Long planId);

    void saveApprovedPlan(DomOrder order, FulfillmentPlan plan);

    DomOrder cancelDomOrder(Long orderId);

    void saveCancelledOrder(DomOrder order);
}
