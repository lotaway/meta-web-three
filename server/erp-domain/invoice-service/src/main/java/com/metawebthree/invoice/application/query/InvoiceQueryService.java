package com.metawebthree.invoice.application.query;

import com.metawebthree.invoice.domain.entity.Invoice;
import com.metawebthree.invoice.domain.repository.InvoiceRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
public class InvoiceQueryService {
    private final InvoiceRepository repository;

    public InvoiceQueryService(InvoiceRepository repository) {
        this.repository = repository;
    }

    public Optional<Invoice> getById(Long id) {
        return repository.findById(id);
    }

    public Optional<Invoice> getByInvoiceNo(String invoiceNo) {
        return repository.findByInvoiceNo(invoiceNo);
    }

    public List<Invoice> listByStatus(String status) {
        Invoice.InvoiceStatus s = Invoice.InvoiceStatus.valueOf(status.toUpperCase());
        return repository.findByStatus(s);
    }

    public List<Invoice> listByCustomerId(Long customerId) {
        return repository.findByCustomerId(customerId);
    }

    public List<Invoice> listByDateRange(LocalDateTime start, LocalDateTime end) {
        return repository.findByDateRange(start, end);
    }

    public List<Invoice> listAll() {
        return repository.findAll();
    }
}