package com.metawebthree.reporting.job;

import com.metawebthree.reporting.application.ReportDeliveryService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@Slf4j
public class ReportSubscriptionJob {

    private final ReportDeliveryService deliveryService;

    @Scheduled(cron = "0 */5 * * * ?")
    public void processSubscriptions() {
        log.info("开始执行报表订阅定时任务");
        try {
            deliveryService.processDueSubscriptions();
        } catch (Exception e) {
            log.error("报表订阅任务执行失败: {}", e.getMessage(), e);
        }
    }
}