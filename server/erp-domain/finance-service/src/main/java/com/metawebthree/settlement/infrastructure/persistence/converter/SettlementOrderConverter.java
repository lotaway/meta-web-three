package com.metawebthree.settlement.infrastructure.persistence.converter;

import com.metawebthree.settlement.domain.entity.SettlementOrder;
import com.metawebthree.settlement.infrastructure.persistence.dataobject.SettlementOrderDO;
import org.springframework.stereotype.Component;

@Component
public class SettlementOrderConverter {

    public SettlementOrder toEntity(SettlementOrderDO doObj) {
        if (doObj == null) return null;
        SettlementOrder entity = new SettlementOrder();
        entity.setId(doObj.getId());
        entity.setSettlementNo(doObj.getSettlementNo());
        entity.setOrderNo(doObj.getOrderNo());
        entity.setMerchantId(doObj.getMerchantId());
        entity.setMerchantName(doObj.getMerchantName());
        entity.setOrderAmount(doObj.getOrderAmount());
        entity.setSettlementAmount(doObj.getSettlementAmount());
        entity.setCommissionAmount(doObj.getCommissionAmount());
        entity.setRefundAmount(doObj.getRefundAmount());
        entity.setStatus(SettlementOrder.SettlementStatus.valueOf(doObj.getStatus()));
        entity.setChannel(doObj.getChannel());
        entity.setSettlementDate(doObj.getSettlementDate());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        entity.setDescription(doObj.getDescription());
        return entity;
    }

    public SettlementOrderDO toDO(SettlementOrder entity) {
        if (entity == null) return null;
        SettlementOrderDO doObj = new SettlementOrderDO();
        doObj.setId(entity.getId());
        doObj.setSettlementNo(entity.getSettlementNo());
        doObj.setOrderNo(entity.getOrderNo());
        doObj.setMerchantId(entity.getMerchantId());
        doObj.setMerchantName(entity.getMerchantName());
        doObj.setOrderAmount(entity.getOrderAmount());
        doObj.setSettlementAmount(entity.getSettlementAmount());
        doObj.setCommissionAmount(entity.getCommissionAmount());
        doObj.setRefundAmount(entity.getRefundAmount());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setChannel(entity.getChannel());
        doObj.setSettlementDate(entity.getSettlementDate());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        doObj.setDescription(entity.getDescription());
        return doObj;
    }
}