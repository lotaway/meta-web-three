package com.metawebthree.rma.infrastructure.job;

import com.metawebthree.rma.application.RmaStockInService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Periodically re-delivers RMA stock-in records that have not reached the
 * inventory service yet, so a transient integration failure recovers on its own.
 */
@Slf4j
@Component
public class RmaStockInRetryJob {

    private final RmaStockInService rmaStockInService;

    @Value("${rma.stock-in.max-retries:10}")
    private int maxRetries;

    public RmaStockInRetryJob(RmaStockInService rmaStockInService) {
        this.rmaStockInService = rmaStockInService;
    }

    @Scheduled(fixedDelayString = "${rma.stock-in.retry-delay-ms:60000}",
            initialDelayString = "${rma.stock-in.retry-initial-delay-ms:30000}")
    public void retryPendingStockIns() {
        try {
            int retried = rmaStockInService.retryPending(maxRetries);
            if (retried > 0) {
                log.info("Retried {} pending RMA stock-in record(s)", retried);
            }
        } catch (Exception e) {
            log.error("RMA stock-in retry job failed", e);
        }
    }
}
