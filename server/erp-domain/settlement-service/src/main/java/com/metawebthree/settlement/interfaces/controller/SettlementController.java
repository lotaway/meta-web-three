package com.metawebthree.settlement.interfaces.controller;

import com.metawebthree.settlement.application.command.SettlementCommandService;
import com.metawebthree.settlement.application.query.SettlementQueryService;
import com.metawebthree.settlement.domain.entity.SettlementOrder;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/settlement")
public class SettlementController {
    private final SettlementCommandService commandService;
    private final SettlementQueryService queryService;

    public SettlementController(SettlementCommandService commandService, SettlementQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping
    public ResponseEntity<Long> create(@RequestBody SettlementRequest request) {
        Long id = commandService.createSettlement(request.getSettlementNo(), request.getOrderNo(),
            request.getMerchantId(), request.getMerchantName(), request.getOrderAmount(), request.getCommissionRate());
        return ResponseEntity.ok(id);
    }

    @GetMapping("/{id}")
    public ResponseEntity<SettlementOrder> get(@PathVariable Long id) {
        return queryService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping
    public ResponseEntity<List<SettlementOrder>> list(@RequestParam(required = false) String status,
                                                       @RequestParam(required = false) Long merchantId,
                                                       @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
                                                       @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        List<SettlementOrder> orders;
        if (status != null) {
            orders = queryService.listByStatus(status);
        } else if (merchantId != null) {
            orders = queryService.listByMerchantId(merchantId);
        } else if (startDate != null && endDate != null) {
            orders = queryService.listByDateRange(startDate, endDate);
        } else {
            orders = queryService.listAll();
        }
        return ResponseEntity.ok(orders);
    }

    @PostMapping("/{id}/confirm")
    public ResponseEntity<Void> confirm(@PathVariable Long id) {
        commandService.confirm(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/process")
    public ResponseEntity<Void> process(@PathVariable Long id) {
        commandService.process(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/complete")
    public ResponseEntity<Void> complete(@PathVariable Long id) {
        commandService.complete(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/fail")
    public ResponseEntity<Void> fail(@PathVariable Long id, @RequestParam String reason) {
        commandService.fail(id, reason);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/cancel")
    public ResponseEntity<Void> cancel(@PathVariable Long id) {
        commandService.cancel(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/refund")
    public ResponseEntity<Void> refund(@PathVariable Long id, @RequestParam BigDecimal amount) {
        commandService.applyRefund(id, amount);
        return ResponseEntity.ok().build();
    }

    public static class SettlementRequest {
        private String settlementNo;
        private String orderNo;
        private Long merchantId;
        private String merchantName;
        private BigDecimal orderAmount;
        private BigDecimal commissionRate;

        public String getSettlementNo() { return settlementNo; }
        public void setSettlementNo(String settlementNo) { this.settlementNo = settlementNo; }
        public String getOrderNo() { return orderNo; }
        public void setOrderNo(String orderNo) { this.orderNo = orderNo; }
        public Long getMerchantId() { return merchantId; }
        public void setMerchantId(Long merchantId) { this.merchantId = merchantId; }
        public String getMerchantName() { return merchantName; }
        public void setMerchantName(String merchantName) { this.merchantName = merchantName; }
        public BigDecimal getOrderAmount() { return orderAmount; }
        public void setOrderAmount(BigDecimal orderAmount) { this.orderAmount = orderAmount; }
        public BigDecimal getCommissionRate() { return commissionRate; }
        public void setCommissionRate(BigDecimal commissionRate) { this.commissionRate = commissionRate; }
    }
}