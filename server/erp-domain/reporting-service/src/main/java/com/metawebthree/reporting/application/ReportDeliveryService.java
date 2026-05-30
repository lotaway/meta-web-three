package com.metawebthree.reporting.application;

import com.metawebthree.reporting.domain.entity.ReportSubscription;
import com.metawebthree.reporting.domain.entity.ReportSubscription.Channel;
import com.metawebthree.reporting.domain.entity.ReportSubscription.ReportType;
import com.metawebthree.reporting.domain.service.ReportSubscriptionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * 报表发送服务
 * 负责生成报表内容并通过不同渠道发送
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ReportDeliveryService {

    private final ReportSubscriptionService subscriptionService;
    private final RestTemplate restTemplate;

    @Value("${report.email.from:noreply@metawebthree.com}")
    private String emailFrom;

    /**
     * 处理所有到期的订阅
     */
    public void processDueSubscriptions() {
        log.info("开始处理到期报表订阅任务");
        var dueSubscriptions = subscriptionService.getDueSubscriptions();
        log.info("发现 {} 个到期订阅", dueSubscriptions.size());
        
        for (var subscription : dueSubscriptions) {
            try {
                processSubscription(subscription);
            } catch (Exception e) {
                log.error("处理订阅 {} 失败: {}", subscription.getId(), e.getMessage());
            }
        }
    }

    /**
     * 处理单个订阅
     */
    private void processSubscription(ReportSubscription subscription) {
        log.info("处理订阅: 用户={}, 类型={}, 频率={}", 
                subscription.getUserName(), 
                subscription.getReportType(),
                subscription.getFrequency());
        
        // 生成报表内容
        String reportContent = generateReportContent(subscription.getReportType());
        
        // 根据渠道发送
        if (subscription.getChannel() == Channel.EMAIL) {
            sendEmail(subscription, reportContent);
        } else if (subscription.getChannel() == Channel.DINGTALK) {
            sendDingTalk(subscription, reportContent);
        }
        
        // 标记已发送并更新下次发送时间
        subscriptionService.markAsSent(subscription.getId());
        log.info("订阅 {} 发送成功", subscription.getId());
    }

    /**
     * 生成报表内容
     */
    private String generateReportContent(ReportType reportType) {
        String dateStr = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"));
        StringBuilder content = new StringBuilder();
        content.append("报表生成时间: ").append(dateStr).append("\n\n");
        
        switch (reportType) {
            case SALES:
                content.append("【销售报表】\n");
                content.append("- 本期销售额: 待统计\n");
                content.append("- 订单数量: 待统计\n");
                content.append("- 客户数量: 待统计\n");
                break;
            case INVENTORY:
                content.append("【库存报表】\n");
                content.append("- 库存总量: 待统计\n");
                content.append("- 库存周转率: 待统计\n");
                content.append("- 缺货商品数: 待统计\n");
                break;
            case FINANCIAL:
                content.append("【财务报表】\n");
                content.append("- 应收总额: 待统计\n");
                content.append("- 应付总额: 待统计\n");
                content.append("- 净利润: 待统计\n");
                break;
        }
        
        content.append("\n如需查看详细报表，请登录系统。");
        return content.toString();
    }

    /**
     * 发送邮件
     */
    private void sendEmail(ReportSubscription subscription, String content) {
        log.info("发送邮件到: {}", subscription.getRecipient());
        // TODO: 集成真实邮件发送服务
        // 实际实现需要调用邮件服务 API
        log.info("邮件内容: {}", content);
    }

    /**
     * 发送钉钉消息
     */
    private void sendDingTalk(ReportSubscription subscription, String content) {
        log.info("发送钉钉消息到 webhook: {}", subscription.getWebhookUrl());
        
        try {
            // 构建钉钉消息体
            String message = buildDingTalkMessage(subscription.getReportType(), content);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<String> entity = new HttpEntity<>(message, headers);
            
            ResponseEntity<String> response = restTemplate.postForEntity(
                    subscription.getWebhookUrl(),
                    entity,
                    String.class
            );
            
            if (response.getStatusCode() == HttpStatus.OK) {
                log.info("钉钉消息发送成功");
            } else {
                log.warn("钉钉消息发送失败: {}", response.getStatusCode());
            }
        } catch (Exception e) {
            log.error("钉钉消息发送失败: {}", e.getMessage());
        }
    }

    /**
     * 构建钉钉消息
     */
    private String buildDingTalkMessage(ReportType reportType, String content) {
        String title = switch (reportType) {
            case SALES -> "销售报表";
            case INVENTORY -> "库存报表";
            case FINANCIAL -> "财务报表";
        };
        
        return String.format(
                "{\"msgtype\": \"text\", \"text\": {\"content\": \"[%s] %s\"}}",
                title,
                content.replace("\n", "\\n")
        );
    }
}