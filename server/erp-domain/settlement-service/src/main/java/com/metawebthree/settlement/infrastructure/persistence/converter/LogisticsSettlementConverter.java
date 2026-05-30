package com.metawebthree.settlement.infrastructure.persistence.converter;

import com.metawebthree.settlement.domain.entity.LogisticsSettlement;
import com.metawebthree.settlement.infrastructure.persistence.dataobject.LogisticsSettlementDO;
import org.springframework.stereotype.Component;

@Component
public class LogisticsSettlementConverter {
    
    public LogisticsSettlement toEntity(LogisticsSettlementDO doObj) {
        if (doObj == null) {
            return null;
        }
        LogisticsSettlement entity = new LogisticsSettlement();
        entity.setId(doObj.getId());
        entity.setSettlementNo(doObj.getSettlementNo());
        entity.setTrackingNo(doObj.getTrackingNo());
        entity.setOrderNo(doObj.getOrderNo());
        entity.setCarrierId(doObj.getCarrierId());
        entity.setCarrierName(doObj.getCarrierName());
        entity.setFreight(doObj.getFreight());
        entity.setHandlingFee(doObj.getHandlingFee());
        entity.setDiscount(doObj.getDiscount());
        entity.setTotalAmount(doObj.getTotalAmount());
        entity.setStatus(LogisticsSettlement.LogisticsSettlementStatus.valueOf(doObj.getStatus()));
        entity.setBillingCycle(doObj.getBillingCycle());
        entity.setSettlementDate(doObj.getSettlementDate());
        entity.setPaidAt(doObj.getPaidAt());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }
    
    public LogisticsSettlementDO toDO(LogisticsSettlement entity) {
        if (entity == null) {
            return null;
        }
        LogisticsSettlementDO doObj = new LogisticsSettlementDO();
        doObj.setId(entity.getId());
        doObj.setSettlementNo(entity.getSettlementNo());
        doObj.setTrackingNo(entity.getTrackingNo());
        doObj.setOrderNo(entity.getOrderNo());
        doObj.setCarrierId(entity.getCarrierId());
        doObj.setCarrierName(entity.getCarrierName());
        doObj.setFreight(entity.getFreight());
        doObj.setHandlingFee(entity.getHandlingFee());
        doObj.setDiscount(entity.getDiscount());
        doObj.setTotalAmount(entity.getTotalAmount());
        doObj.setStatus(entity.getStatus().name());
        doObj.setBillingCycle(entity.getBillingCycle());
        doObj.setSettlementDate(entity.getSettlementDate());
        doObj.setPaidAt(entity.getPaidAt());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}