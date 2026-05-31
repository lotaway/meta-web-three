package com.metawebthree.order.job;

import com.metawebthree.order.application.OrderApplicationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "order.auto.cancel.enabled", havingValue = "true", matchIfMissing = true)
public class OrderAutoCancelJob {

    private final OrderApplicationService orderApplicationService;

    @Scheduled(cron = "${order.auto.cancel.cron:0 */5 * * * ?}")
    public void cancelTimeOutOrders() {
        log.info("Starting timeout order auto-cancel job");
        try {
            int defaultTimeoutMinutes = 30;
            int count = orderApplicationService.cancelTimeOutOrder(defaultTimeoutMinutes);
            log.info("Timeout order auto-cancel job completed, cancelled order count: {}", count);
        } catch (Exception e) {
            log.error("Timeout order auto-cancel job execution failed", e);
        }
    }
}