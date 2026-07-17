package com.metawebthree.dom.application;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.metawebthree.dom.application.dto.*;
import java.util.List;

public interface DomApplicationService {

    DomOrderDTO createDomOrder(CreateDomOrderRequest request);

    DomOrderDTO getDomOrder(Long id);

    DomOrderDTO getDomOrderByNo(String domOrderNo);

    IPage<DomOrderDTO> listDomOrders(DomQueryParam param);

    DomOrderDTO checkAvailability(Long orderId);

    DomOrderDTO sourceOrder(Long orderId);

    FulfillmentPlanDTO approveFulfillment(Long orderId);

    DomOrderDTO cancelDomOrder(Long orderId);

    List<SourcingRuleDTO> getSourcingRules();

    SourcingRuleDTO updateSourcingRule(SourcingRuleDTO rule);

    SourcingRuleDTO createSourcingRule(SourcingRuleDTO rule);

    void deleteSourcingRule(Long id);
}
