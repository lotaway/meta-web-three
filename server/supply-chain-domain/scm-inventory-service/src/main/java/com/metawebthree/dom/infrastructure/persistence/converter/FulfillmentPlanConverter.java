package com.metawebthree.dom.infrastructure.persistence.converter;

import com.metawebthree.dom.domain.entity.FulfillmentPlan;
import com.metawebthree.dom.domain.entity.FulfillmentPlanStatus;
import com.metawebthree.dom.infrastructure.persistence.dataobject.FulfillmentPlanDO;
import org.springframework.stereotype.Component;

@Component
public class FulfillmentPlanConverter {

    public FulfillmentPlan toEntity(FulfillmentPlanDO doObj) {
        if (doObj == null) {
            return null;
        }
        FulfillmentPlan entity = new FulfillmentPlan();
        entity.setId(doObj.getId());
        entity.setDomOrderId(doObj.getDomOrderId());
        entity.setDomOrderNo(doObj.getDomOrderNo());
        entity.setTotalLines(doObj.getTotalLines());
        entity.setFulfilledLines(doObj.getFulfilledLines());
        entity.setPartiallyFulfilledLines(doObj.getPartiallyFulfilledLines());
        entity.setUnfulfilledLines(doObj.getUnfulfilledLines());
        entity.setStatus(doObj.getStatus() != null ? FulfillmentPlanStatus.valueOf(doObj.getStatus()) : null);
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public FulfillmentPlanDO toDO(FulfillmentPlan entity) {
        if (entity == null) {
            return null;
        }
        FulfillmentPlanDO doObj = new FulfillmentPlanDO();
        doObj.setId(entity.getId());
        doObj.setDomOrderId(entity.getDomOrderId());
        doObj.setDomOrderNo(entity.getDomOrderNo());
        doObj.setTotalLines(entity.getTotalLines());
        doObj.setFulfilledLines(entity.getFulfilledLines());
        doObj.setPartiallyFulfilledLines(entity.getPartiallyFulfilledLines());
        doObj.setUnfulfilledLines(entity.getUnfulfilledLines());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
