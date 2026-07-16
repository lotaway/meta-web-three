package com.metawebthree.rma.infrastructure.persistence.converter;

import com.metawebthree.rma.domain.entity.RmaInspection;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaInspectionDO;
import org.springframework.stereotype.Component;

@Component
public class RmaInspectionConverter {

    public RmaInspection toEntity(RmaInspectionDO doObj) {
        if (doObj == null) {
            return null;
        }
        RmaInspection entity = new RmaInspection();
        entity.setId(doObj.getId());
        entity.setRmaId(doObj.getRmaId());
        entity.setRmaNo(doObj.getRmaNo());
        entity.setInspector(doObj.getInspector());
        entity.setInspectionDate(doObj.getInspectionDate());
        entity.setResult(doObj.getResult());
        entity.setConclusion(doObj.getConclusion());
        entity.setTotalInspected(doObj.getTotalInspected());
        entity.setTotalPassed(doObj.getTotalPassed());
        entity.setTotalFailed(doObj.getTotalFailed());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public RmaInspectionDO toDO(RmaInspection entity) {
        if (entity == null) {
            return null;
        }
        RmaInspectionDO doObj = new RmaInspectionDO();
        doObj.setId(entity.getId());
        doObj.setRmaId(entity.getRmaId());
        doObj.setRmaNo(entity.getRmaNo());
        doObj.setInspector(entity.getInspector());
        doObj.setInspectionDate(entity.getInspectionDate());
        doObj.setResult(entity.getResult());
        doObj.setConclusion(entity.getConclusion());
        doObj.setTotalInspected(entity.getTotalInspected());
        doObj.setTotalPassed(entity.getTotalPassed());
        doObj.setTotalFailed(entity.getTotalFailed());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
