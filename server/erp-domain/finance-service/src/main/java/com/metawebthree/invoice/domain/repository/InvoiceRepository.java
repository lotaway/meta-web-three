package com.metawebthree.invoice.domain.repository;

import com.metawebthree.invoice.domain.entity.Invoice;
import java.util.List;
import java.util.Optional;

public interface InvoiceRepository {
    Optional<Invoice> findById(Long id);
    Optional<Invoice> findByInvoiceNo(String invoiceNo);
    List<Invoice> findByStatus(Invoice.InvoiceStatus status);
    List<Invoice> findByCustomerId(Long customerId);
    List<Invoice> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end);
    List<Invoice> findAll();
    void save(Invoice invoice);
    void update(Invoice invoice);
    void delete(Long id);
}