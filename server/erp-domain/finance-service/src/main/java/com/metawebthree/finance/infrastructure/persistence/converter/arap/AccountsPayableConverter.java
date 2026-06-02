package com.metawebthree.finance.infrastructure.persistence.converter.arap;

import com.metawebthree.finance.domain.entity.arap.AccountsPayable;
import com.metawebthree.finance.infrastructure.persistence.dataobject.arap.AccountsPayableDO;
import org.springframework.stereotype.Component;

@Component
public class AccountsPayableConverter {

    public AccountsPayable toEntity(AccountsPayableDO doObj) {
        if (doObj == null) {
            return null;
        }
        AccountsPayable entity = new AccountsPayable();
        entity.setId(doObj.getId());
        entity.setApCode(doObj.getApCode());
        entity.setSupplierId(doObj.getSupplierId());
        entity.setSupplierName(doObj.getSupplierName());
        entity.setBusinessType(doObj.getBusinessType());
        entity.setRelatedDocumentType(doObj.getRelatedDocumentType());
        entity.setRelatedDocumentNo(doObj.getRelatedDocumentNo());
        entity.setAmount(doObj.getAmount());
        entity.setPaidAmount(doObj.getPaidAmount());
        entity.setRemainingAmount(doObj.getRemainingAmount());
        entity.setInvoiceDate(doObj.getInvoiceDate());
        entity.setDueDate(doObj.getDueDate());
        entity.setCreditTerm(doObj.getCreditTerm());
        entity.setStatus(doObj.getStatus() != null ? AccountsPayable.ApStatus.valueOf(doObj.getStatus()) : null);
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

    public AccountsPayableDO toDO(AccountsPayable entity) {
        if (entity == null) {
            return null;
        }
        AccountsPayableDO doObj = new AccountsPayableDO();
        doObj.setId(entity.getId());
        doObj.setApCode(entity.getApCode());
        doObj.setSupplierId(entity.getSupplierId());
        doObj.setSupplierName(entity.getSupplierName());
        doObj.setBusinessType(entity.getBusinessType());
        doObj.setRelatedDocumentType(entity.getRelatedDocumentType());
        doObj.setRelatedDocumentNo(entity.getRelatedDocumentNo());
        doObj.setAmount(entity.getAmount());
        doObj.setPaidAmount(entity.getPaidAmount());
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