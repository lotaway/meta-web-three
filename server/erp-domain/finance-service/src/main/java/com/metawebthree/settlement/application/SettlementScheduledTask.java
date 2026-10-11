package com.metawebthree.settlement.application;

import com.metawebthree.settlement.domain.entity.SettlementOrder;
import com.metawebthree.settlement.domain.repository.SettlementOrderRepository;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import java.time.LocalDateTime;
import java.util.List;

@Component
public class SettlementScheduledTask {
    private final SettlementOrderRepository repository;

    public SettlementScheduledTask(SettlementOrderRepository repository) {
        this.repository = repository;
    }

    @Scheduled(cron = "0 0 2 * * ?")
    public void dailySettlement() {
        List<SettlementOrder> pendingSettlements = repository.findByStatus(SettlementOrder.SettlementStatus.CONFIRMED);
        for (SettlementOrder order : pendingSettlements) {
            try {
                order.process();
                order.complete();
                repository.update(order);
            } catch (Exception e) {
                order.fail("Auto settlement failed: " + e.getMessage());
                repository.update(order);
            }
        }
    }

    @Scheduled(cron = "0 0 0 1 * ?")
    public void monthlyReconciliation() {
        LocalDateTime lastMonth = LocalDateTime.now().minusMonths(1);
        LocalDateTime startOfLastMonth = lastMonth.withDayOfMonth(1).withHour(0).withMinute(0).withSecond(0);
        LocalDateTime endOfLastMonth = lastMonth.withDayOfMonth(lastMonth.toLocalDate().lengthOfMonth())
            .withHour(23).withMinute(59).withSecond(59);
        List<SettlementOrder> monthlySettlements = repository.findByDateRange(startOfLastMonth, endOfLastMonth);
        // 触发对账流程
    }
}