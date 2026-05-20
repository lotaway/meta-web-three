package com.metawebthree.invoice.application.command;

import com.metawebthree.invoice.domain.entity.Invoice;
import com.metawebthree.invoice.domain.repository.InvoiceRepository;
import com.metawebthree.invoice.infrastructure.event.InvoiceEventPublisher;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;

@Service
public class InvoiceCommandService {
    private final InvoiceRepository repository;
    private final InvoiceEventPublisher eventPublisher;

    public InvoiceCommandService(InvoiceRepository repository, InvoiceEventPublisher eventPublisher) {
        this.repository = repository;
        this.eventPublisher = eventPublisher;
    }

    public Long createInvoice(String invoiceNo, String orderNo, Long customerId, String customerName,
                              String customerTaxNo, String type, BigDecimal amount, String taxRate) {
        Invoice.InvoiceType invoiceType = Invoice.InvoiceType.valueOf(type.toUpperCase());
        Invoice invoice = new Invoice();
        invoice.createDraft(invoiceNo, orderNo, customerId, customerName, customerTaxNo, invoiceType, amount, taxRate);
        repository.save(invoice);
        eventPublisher.publishInvoiceCreated(invoice.getId(), invoiceNo, customerId, amount);
        return invoice.getId();
    }

    public void issue(Long invoiceId, String issuer) {
        Invoice invoice = repository.findById(invoiceId)
            .orElseThrow(() -> new IllegalArgumentException("Invoice not found"));
        invoice.issue(issuer);
        repository.update(invoice);
        eventPublisher.publishInvoiceIssued(invoiceId, invoice.getInvoiceNo());
    }

    public void print(Long invoiceId) {
        Invoice invoice = repository.findById(invoiceId)
            .orElseThrow(() -> new IllegalArgumentException("Invoice not found"));
        invoice.print();
        repository.update(invoice);
    }

    public void voidInvoice(Long invoiceId, String reason) {
        Invoice invoice = repository.findById(invoiceId)
            .orElseThrow(() -> new IllegalArgumentException("Invoice not found"));
        invoice.voidInvoice(reason);
        repository.update(invoice);
        eventPublisher.publishInvoiceVoided(invoiceId, invoice.getInvoiceNo(), reason);
    }

    public void redFlush(Long invoiceId, String reason) {
        Invoice invoice = repository.findById(invoiceId)
            .orElseThrow(() -> new IllegalArgumentException("Invoice not found"));
        invoice.redFlush(reason);
        repository.update(invoice);
        eventPublisher.publishInvoiceRedFlushed(invoiceId, invoice.getInvoiceNo(), reason);
    }

    public void updateCustomerInfo(Long invoiceId, String name, String taxNo, String address, 
                                    String bank, String account) {
        Invoice invoice = repository.findById(invoiceId)
            .orElseThrow(() -> new IllegalArgumentException("Invoice not found"));
        invoice.updateCustomerInfo(name, taxNo, address, bank, account);
        repository.update(invoice);
    }
}