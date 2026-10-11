package com.metawebthree.payment.job;

import com.metawebthree.payment.application.ReconciliationServiceImpl;
import com.metawebthree.payment.application.ReconciliationReportService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.quartz.Job;
import org.quartz.JobExecutionContext;
import org.quartz.JobExecutionException;
import org.springframework.stereotype.Component;

import java.time.LocalDate;

@Component
@RequiredArgsConstructor
@Slf4j
public class ReconciliationDailyJob implements Job {

    private final ReconciliationServiceImpl reconciliationService;
    private final ReconciliationReportService reconciliationReportService;

    @Override
    public void execute(JobExecutionContext context) throws JobExecutionException {
        LocalDate targetDate = getReconciliationDate();
        log.info("Starting daily reconciliation job for date: {}", targetDate);
        
        try {
            runReconciliation(targetDate);
            log.info("Daily reconciliation job completed successfully");
        } catch (Exception e) {
            log.error("Daily reconciliation job failed", e);
            throw new JobExecutionException(e);
        }
    }

    private LocalDate getReconciliationDate() {
        return LocalDate.now().minusDays(1);
    }

    private void runReconciliation(LocalDate targetDate) {
        reconciliationService.executeDailyReconciliation();
        reconciliationReportService.generateDailyReport(targetDate);
    }
}