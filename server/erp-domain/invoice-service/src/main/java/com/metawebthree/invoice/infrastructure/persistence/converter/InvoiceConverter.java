package com.metawebthree.invoice.infrastructure.persistence.converter;

import com.metawebthree.invoice.domain.entity.Invoice;
import com.metawebthree.invoice.infrastructure.persistence.dataobject.InvoiceDO;
import org.springframework.stereotype.Component;

@Component
public class InvoiceConverter {

    public Invoice toEntity(InvoiceDO doObj) {
        if (doObj == null) {
            return null;
        }
        Invoice entity = new Invoice();
        entity.setId(doObj.getId());
        entity.setInvoiceNo(doObj.getInvoiceNo());
        entity.setOrderNo(doObj.getOrderNo());
        entity.setCustomerId(doObj.getCustomerId());
        entity.setCustomerName(doObj.getCustomerName());
        entity.setCustomerTaxNo(doObj.getCustomerTaxNo());
        entity.setCustomerAddress(doObj.getCustomerAddress());
        entity.setCustomerBank(doObj.getCustomerBank());
        entity.setCustomerAccount(doObj.getCustomerAccount());
        entity.setType(Invoice.InvoiceType.valueOf(doObj.getType()));
        entity.setStatus(Invoice.InvoiceStatus.valueOf(doObj.getStatus()));
        entity.setAmount(doObj.getAmount());
        entity.setTaxAmount(doObj.getTaxAmount());
        entity.setTotalAmount(doObj.getTotalAmount());
        entity.setTaxRate(doObj.getTaxRate());
        entity.setIssueDate(doObj.getIssueDate());
        entity.setIssuer(doObj.getIssuer());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public InvoiceDO toDO(Invoice entity) {
        if (entity == null) {
            return null;
        }
        InvoiceDO doObj = new InvoiceDO();
        doObj.setId(entity.getId());
        doObj.setInvoiceNo(entity.getInvoiceNo());
        doObj.setOrderNo(entity.getOrderNo());
        doObj.setCustomerId(entity.getCustomerId());
        doObj.setCustomerName(entity.getCustomerName());
        doObj.setCustomerTaxNo(entity.getCustomerTaxNo());
        doObj.setCustomerAddress(entity.getCustomerAddress());
        doObj.setCustomerBank(entity.getCustomerBank());
        doObj.setCustomerAccount(entity.getCustomerAccount());
        doObj.setType(entity.getType() != null ? entity.getType().name() : null);
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setAmount(entity.getAmount());
        doObj.setTaxAmount(entity.getTaxAmount());
        doObj.setTotalAmount(entity.getTotalAmount());
        doObj.setTaxRate(entity.getTaxRate());
        doObj.setIssueDate(entity.getIssueDate());
        doObj.setIssuer(entity.getIssuer());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}