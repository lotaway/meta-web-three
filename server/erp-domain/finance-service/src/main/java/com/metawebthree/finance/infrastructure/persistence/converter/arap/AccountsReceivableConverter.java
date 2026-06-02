package com.metawebthree.finance.infrastructure.persistence.converter.arap;

import com.metawebthree.finance.domain.entity.arap.AccountsReceivable;
import com.metawebthree.finance.infrastructure.persistence.dataobject.arap.AccountsReceivableDO;
import org.springframework.stereotype.Component;

@Component
public class AccountsReceivableConverter {

    public AccountsReceivable toEntity(AccountsReceivableDO doObj) {
        if (doObj == null) {
            return null;
        }
        AccountsReceivable entity = new AccountsReceivable();
        entity.setId(doObj.getId());
        entity.setArCode(doObj.getArCode());
        entity.setCustomerId(doObj.getCustomerId());
        entity.setCustomerName(doObj.getCustomerName());
        entity.setBusinessType(doObj.getBusinessType());
        entity.setRelatedDocumentType(doObj.getRelatedDocumentType());
        entity.setRelatedDocumentNo(doObj.getRelatedDocumentNo());
        entity.setAmount(doObj.getAmount());
        entity.setReceivedAmount(doObj.getReceivedAmount());
        entity.setRemainingAmount(doObj.getRemainingAmount());
        entity.setInvoiceDate(doObj.getInvoiceDate());
        entity.setDueDate(doObj.getDueDate());
        entity.setCreditTerm(doObj.getCreditTerm());
        entity.setStatus(doObj.getStatus() != null ? AccountsReceivable.ArStatus.valueOf(doObj.getStatus()) : null);
        entity.setCurrency(doObj.getCurrency());
        entity.setExchangeRate(doObj.getExchangeRate());
        entity.setOriginalAmount(doObj.getOriginalAmount());
        entity.setDescription(doObj.getDescription());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatorName(doObj.getCreatorName());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        entity.setIsActive(doObj.getIsActive());
        return entity;
    }

    public AccountsReceivableDO toDO(AccountsReceivable entity) {
        if (entity == null) {
            return null;
        }
        AccountsReceivableDO doObj = new AccountsReceivableDO();
        doObj.setId(entity.getId());
        doObj.setArCode(entity.getArCode());
        doObj.setCustomerId(entity.getCustomerId());
        doObj.setCustomerName(entity.getCustomerName());
        doObj.setBusinessType(entity.getBusinessType());
        doObj.setRelatedDocumentType(entity.getRelatedDocumentType());
        doObj.setRelatedDocumentNo(entity.getRelatedDocumentNo());
        doObj.setAmount(entity.getAmount());
        doObj.setReceivedAmount(entity.getReceivedAmount());
        doObj.setRemainingAmount(entity.getRemainingAmount());
        doObj.setInvoiceDate(entity.getInvoiceDate());
        doObj.setDueDate(entity.getDueDate());
        doObj.setCreditTerm(entity.getCreditTerm());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setCurrency(entity.getCurrency());
        doObj.setExchangeRate(entity.getExchangeRate());
        doObj.setOriginalAmount(entity.getOriginalAmount());
        doObj.setDescription(entity.getDescription());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatorName(entity.getCreatorName());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        doObj.setIsActive(entity.getIsActive());
        return doObj;
    }
}