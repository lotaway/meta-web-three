package com.metawebthree.commission.application;

import java.time.LocalDateTime;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class CommissionSettlementJob {
    private final CommissionCommandService commandService;
    private final TimeProvider timeProvider;

    public CommissionSettlementJob(CommissionCommandService commandService, TimeProvider timeProvider) {
        this.commandService = commandService;
        this.timeProvider = timeProvider;
    }

    @Scheduled(cron = "${commission.settlement.cron}")
    public void settleDueRecords() {
        LocalDateTime now = timeProvider.now();
        commandService.settleBefore(now);
    }
}
