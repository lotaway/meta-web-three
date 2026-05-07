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
        log.info("开始执行超时订单自动取消任务");
        try {
            int defaultTimeoutMinutes = 30;
            int count = orderApplicationService.cancelTimeOutOrder(defaultTimeoutMinutes);
            log.info("超时订单自动取消任务完成，取消订单数: {}", count);
        } catch (Exception e) {
            log.error("超时订单自动取消任务执行失败", e);
        }
    }
}