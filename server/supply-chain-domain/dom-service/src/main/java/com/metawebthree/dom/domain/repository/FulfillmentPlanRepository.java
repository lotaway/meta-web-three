package com.metawebthree.dom.domain.repository;

import com.metawebthree.dom.domain.entity.FulfillmentPlan;
import java.util.List;
import java.util.Optional;

public interface FulfillmentPlanRepository {

    Optional<FulfillmentPlan> findById(Long id);

    Optional<FulfillmentPlan> findByDomOrderId(Long domOrderId);

    List<FulfillmentPlan> findByStatus(String status);

    FulfillmentPlan save(FulfillmentPlan plan);
}
