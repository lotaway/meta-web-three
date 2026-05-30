package com.metawebthree.reporting.job;

import com.metawebthree.reporting.application.ReportDeliveryService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * 报表订阅定时任务
 * 定时检查并发送到期的报表订阅
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ReportSubscriptionJob {

    private final ReportDeliveryService deliveryService;

    /**
     * 每5分钟执行一次，检查是否有需要发送的报表
     */
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