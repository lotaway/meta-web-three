package com.metawebthree.invoice.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.ERPPermissions;
import com.metawebthree.invoice.application.command.InvoiceCommandService;
import com.metawebthree.invoice.application.query.InvoiceQueryService;
import com.metawebthree.invoice.domain.entity.Invoice;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/invoice")
public class InvoiceController {
    private final InvoiceCommandService commandService;
    private final InvoiceQueryService queryService;

    public InvoiceController(InvoiceCommandService commandService, InvoiceQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @RequirePermission(ERPPermissions.INVOICE_CREATE)
    @PostMapping
    public ResponseEntity<Long> create(@RequestBody InvoiceRequest request,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        Long id = commandService.createInvoice(request.getInvoiceNo(), request.getOrderNo(),
            request.getCustomerId(), request.getCustomerName(), request.getCustomerTaxNo(),
            request.getType(), request.getAmount(), request.getTaxRate());
        return ResponseEntity.ok(id);
    }

    @RequirePermission(ERPPermissions.INVOICE_READ)
    @GetMapping("/{id}")
    public ResponseEntity<Invoice> get(@PathVariable Long id) {
        return queryService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @RequirePermission(ERPPermissions.INVOICE_READ)
    @GetMapping
    public ResponseEntity<List<Invoice>> list(@RequestParam(required = false) String status,
                                               @RequestParam(required = false) Long customerId,
                                               @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
                                               @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        List<Invoice> invoices;
        if (status != null) {
            invoices = queryService.listByStatus(status);
        } else if (customerId != null) {
            invoices = queryService.listByCustomerId(customerId);
        } else if (startDate != null && endDate != null) {
            invoices = queryService.listByDateRange(startDate, endDate);
        } else {
            invoices = queryService.listAll();
        }
        return ResponseEntity.ok(invoices);
    }

    @RequirePermission(ERPPermissions.INVOICE_ISSUE)
    @PostMapping("/{id}/issue")
    public ResponseEntity<Void> issue(@PathVariable Long id, 
            @RequestParam String issuer,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.issue(id, issuer);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.INVOICE_PRINT)
    @PostMapping("/{id}/print")
    public ResponseEntity<Void> print(@PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.print(id);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.INVOICE_VOID)
    @PostMapping("/{id}/void")
    public ResponseEntity<Void> voidInvoice(@PathVariable Long id, 
            @RequestParam String reason,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.voidInvoice(id, reason);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.INVOICE_RED_FLUSH)
    @PostMapping("/{id}/red-flush")
    public ResponseEntity<Void> redFlush(@PathVariable Long id, 
            @RequestParam String reason,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.redFlush(id, reason);
        return ResponseEntity.ok().build();
    }

    public static class InvoiceRequest {
        private String invoiceNo;
        private String orderNo;
        private Long customerId;
        private String customerName;
        private String customerTaxNo;
        private String type;
        private BigDecimal amount;
        private String taxRate;

        public String getInvoiceNo() { return invoiceNo; }
        public void setInvoiceNo(String invoiceNo) { this.invoiceNo = invoiceNo; }
        public String getOrderNo() { return orderNo; }
        public void setOrderNo(String orderNo) { this.orderNo = orderNo; }
        public Long getCustomerId() { return customerId; }
        public void setCustomerId(Long customerId) { this.customerId = customerId; }
        public String getCustomerName() { return customerName; }
        public void setCustomerName(String customerName) { this.customerName = customerName; }
        public String getCustomerTaxNo() { return customerTaxNo; }
        public void setCustomerTaxNo(String customerTaxNo) { this.customerTaxNo = customerTaxNo; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public BigDecimal getAmount() { return amount; }
        public void setAmount(BigDecimal amount) { this.amount = amount; }
        public String getTaxRate() { return taxRate; }
        public void setTaxRate(String taxRate) { this.taxRate = taxRate; }
    }
}