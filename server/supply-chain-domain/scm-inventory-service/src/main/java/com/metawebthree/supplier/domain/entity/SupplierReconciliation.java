package com.metawebthree.supplier.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Data
public class SupplierReconciliation {
    private Long id;
    private String reconciliationNo;
    private String supplierCode;
    private LocalDate periodStart;
    private LocalDate periodEnd;
    private Integer orderCount;
    private BigDecimal totalAmount;
    private BigDecimal shippedAmount;
    private BigDecimal invoicedAmount;
    private BigDecimal settledAmount;
    private BigDecimal pendingAmount;
    private String currency;
    private String status;
    private LocalDateTime submittedAt;
    private LocalDateTime confirmedAt;
    private String confirmedBy;
    private LocalDateTime paidAt;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<SupplierReconciliationItem> items = new ArrayList<>();

    public void submit() {
        if ("PENDING".equals(status)) {
            this.status = "SUBMITTED";
            this.submittedAt = LocalDateTime.now();
        }
    }

    public void confirm(String confirmedBy) {
        if ("SUBMITTED".equals(status)) {
            this.status = "CONFIRMED";
            this.confirmedBy = confirmedBy;
            this.confirmedAt = LocalDateTime.now();
        }
    }

    public void reject(String remark) {
        if ("SUBMITTED".equals(status)) {
            this.status = "REJECTED";
            this.remark = remark;
        }
    }

    public void markAsPaid() {
        if ("CONFIRMED".equals(status)) {
            this.status = "PAID";
            this.paidAt = LocalDateTime.now();
        }
    }

    public void calculateTotals() {
        if (items != null && !items.isEmpty()) {
            this.orderCount = items.size();
            BigDecimal total = BigDecimal.ZERO;
            BigDecimal shipped = BigDecimal.ZERO;
            BigDecimal invoiced = BigDecimal.ZERO;
            BigDecimal settled = BigDecimal.ZERO;
            BigDecimal pending = BigDecimal.ZERO;
            
            for (SupplierReconciliationItem item : items) {
                if (item.getInvoicedAmount() != null) total = total.add(item.getInvoicedAmount());
                if (item.getSettledAmount() != null) shipped = shipped.add(item.getSettledAmount());
                if (item.getInvoicedAmount() != null) invoiced = invoiced.add(item.getInvoicedAmount());
                if (item.getSettledAmount() != null) settled = settled.add(item.getSettledAmount());
                if (item.getPendingAmount() != null) pending = pending.add(item.getPendingAmount());
            }
            
            this.totalAmount = total;
            this.shippedAmount = shipped;
            this.invoicedAmount = invoiced;
            this.settledAmount = settled;
            this.pendingAmount = pending;
        }
    }

    public boolean canSubmit() {
        return "PENDING".equals(status);
    }

    public boolean canConfirm() {
        return "SUBMITTED".equals(status);
    }

    public boolean canReject() {
        return "SUBMITTED".equals(status);
    }

    public boolean canMarkPaid() {
        return "CONFIRMED".equals(status);
    }
}