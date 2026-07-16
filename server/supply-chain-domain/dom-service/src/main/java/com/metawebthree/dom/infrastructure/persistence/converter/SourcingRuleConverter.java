package com.metawebthree.dom.infrastructure.persistence.converter;

import com.metawebthree.dom.domain.entity.SourcingRule;
import com.metawebthree.dom.infrastructure.persistence.dataobject.SourcingRuleDO;
import org.springframework.stereotype.Component;

@Component
public class SourcingRuleConverter {

    public SourcingRule toEntity(SourcingRuleDO doObj) {
        if (doObj == null) {
            return null;
        }
        SourcingRule entity = new SourcingRule();
        entity.setId(doObj.getId());
        entity.setRuleName(doObj.getRuleName());
        entity.setRuleType(doObj.getRuleType());
        entity.setPriority(doObj.getPriority());
        entity.setWarehouseIds(doObj.getWarehouseIds());
        entity.setRegion(doObj.getRegion());
        entity.setEnabled(doObj.getEnabled());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public SourcingRuleDO toDO(SourcingRule entity) {
        if (entity == null) {
            return null;
        }
        SourcingRuleDO doObj = new SourcingRuleDO();
        doObj.setId(entity.getId());
        doObj.setRuleName(entity.getRuleName());
        doObj.setRuleType(entity.getRuleType());
        doObj.setPriority(entity.getPriority());
        doObj.setWarehouseIds(entity.getWarehouseIds());
        doObj.setRegion(entity.getRegion());
        doObj.setEnabled(entity.getEnabled());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
