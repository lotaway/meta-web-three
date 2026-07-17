package com.metawebthree.reporting.application;

import com.metawebthree.reporting.domain.entity.*;
import com.metawebthree.reporting.domain.entity.ReportSubscription.Channel;
import com.metawebthree.reporting.domain.entity.ReportSubscription.ReportType;
import com.metawebthree.reporting.domain.repository.FinancialReportRepository;
import com.metawebthree.reporting.domain.repository.InventoryReportRepository;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import com.metawebthree.reporting.domain.service.ReportSubscriptionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Objects;

@Service
@RequiredArgsConstructor
@Slf4j
public class ReportDeliveryService {

    private final ReportSubscriptionService subscriptionService;
    private final SalesReportRepository salesReportRepository;
    private final InventoryReportRepository inventoryReportRepository;
    private final FinancialReportRepository financialReportRepository;
    private final RestTemplate restTemplate;

    @Value("${report.email.from:noreply@metawebthree.com}")
    private String emailFrom;

    @Value("${report.email.enabled:false}")
    private boolean emailEnabled;

    public void processDueSubscriptions() {
        log.info("开始处理到期报表订阅任务");
        var dueSubscriptions = subscriptionService.getDueSubscriptions();
        log.info("发现 {} 个到期订阅", dueSubscriptions.size());
        
        for (var subscription : dueSubscriptions) {
            try {
                processSubscription(subscription);
            } catch (Exception e) {
                log.error("处理订阅 {} 失败: {}", subscription.getId(), e.getMessage(), e);
            }
        }
    }

    private void processSubscription(ReportSubscription subscription) {
        log.info("处理订阅: 用户={}, 类型={}, 频率={}", 
                subscription.getUserName(), 
                subscription.getReportType(),
                subscription.getFrequency());
        
        String reportContent = generateReportContent(subscription.getReportType());
        
        if (subscription.getChannel() == Channel.EMAIL) {
            sendEmail(subscription, reportContent);
        } else if (subscription.getChannel() == Channel.DINGTALK) {
            sendDingTalk(subscription, reportContent);
        }
        
        subscriptionService.markAsSent(subscription.getId());
        log.info("订阅 {} 发送成功", subscription.getId());
    }

    private String generateReportContent(ReportType reportType) {
        String dateStr = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        StringBuilder content = new StringBuilder();
        content.append("══════════════════════════════════════\n");
        content.append("        ").append(getReportTitle(reportType)).append("\n");
        content.append("══════════════════════════════════════\n");
        content.append("生成时间: ").append(dateStr).append("\n\n");
        
        switch (reportType) {
            case SALES -> appendSalesReportContent(content);
            case INVENTORY -> appendInventoryReportContent(content);
            case FINANCIAL -> appendFinancialReportContent(content);
        }
        
        content.append("\n──────────────────────────────────────\n");
        content.append("如需查看详细报表，请登录系统。");
        return content.toString();
    }
    
    private String getReportTitle(ReportType reportType) {
        return switch (reportType) {
            case SALES -> "销售报表";
            case INVENTORY -> "库存报表";
            case FINANCIAL -> "财务报表";
        };
    }
    
    private void appendSalesReportContent(StringBuilder content) {
        content.append("【本期销售概况】\n");
        
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime dayStart = now.toLocalDate().atStartOfDay();
        LocalDateTime dayEnd = now.toLocalDate().atTime(23, 59, 59);
        
        List<SalesReport> todayReports = salesReportRepository.findByDateRange(dayStart, dayEnd);
        BigDecimal todaySales = todayReports.stream()
            .map(SalesReport::getTotalSalesAmount)
            .filter(Objects::nonNull)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        int todayOrders = todayReports.stream()
            .mapToInt(r -> r.getTotalOrderCount() != null ? r.getTotalOrderCount() : 0)
            .sum();
        
        content.append("  📊 今日销售额: ").append(formatAmount(todaySales)).append(" 元\n");
        content.append("  📦 今日订单数: ").append(todayOrders).append(" 单\n");
        
        LocalDateTime monthStart = now.withDayOfMonth(1).toLocalDate().atStartOfDay();
        List<SalesReport> monthReports = salesReportRepository.findByDateRange(monthStart, dayEnd);
        BigDecimal monthSales = monthReports.stream()
            .map(SalesReport::getTotalSalesAmount)
            .filter(Objects::nonNull)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        int monthOrders = monthReports.stream()
            .mapToInt(r -> r.getTotalOrderCount() != null ? r.getTotalOrderCount() : 0)
            .sum();
        
        content.append("\n【本月累计】\n");
        content.append("  💰 月销售额: ").append(formatAmount(monthSales)).append(" 元\n");
        content.append("  📋 月订单数: ").append(monthOrders).append(" 单\n");
        
        if (monthOrders > 0) {
            BigDecimal avgOrder = monthSales.divide(BigDecimal.valueOf(monthOrders), 2, RoundingMode.HALF_UP);
            content.append("  📈 客单价: ").append(formatAmount(avgOrder)).append(" 元\n");
        }
    }
    
    private void appendInventoryReportContent(StringBuilder content) {
        content.append("【库存概况】\n");
        
        List<InventoryReport> reports = inventoryReportRepository.findAll();
        if (!reports.isEmpty()) {
            InventoryReport latest = reports.get(reports.size() - 1);
            content.append("  📊 库存总量: ").append(latest.getTotalQuantity() != null ? latest.getTotalQuantity() : "-").append(" 件\n");
            content.append("  📦 SKU种类: ").append(latest.getTotalSkuCount() != null ? latest.getTotalSkuCount() : "-").append(" 种\n");
            content.append("  🔄 库存周转率: ").append(latest.getTurnoverRate() != null ? latest.getTurnoverRate() + "%" : "-").append("\n");
            content.append("  📉 滞销商品数: ").append(latest.getSlowMovingCount() != null ? latest.getSlowMovingCount() : "-").append(" 种\n");
        } else {
            content.append("  暂无库存数据\n");
        }
        
        content.append("\n【库存分布】\n");
        content.append("  详细数据请登录系统查看\n");
    }
    
    private void appendFinancialReportContent(StringBuilder content) {
        content.append("【财务概况】\n");
        
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime monthStart = now.withDayOfMonth(1).toLocalDate().atStartOfDay();
        LocalDateTime monthEnd = now.toLocalDate().atTime(23, 59, 59);
        
        List<FinancialReport> reports = financialReportRepository.findByDateRange(monthStart, monthEnd);
        
        BigDecimal totalReceivable = reports.stream()
            .map(FinancialReport::getTotalReceivable)
            .filter(Objects::nonNull)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        BigDecimal totalPayable = reports.stream()
            .map(FinancialReport::getTotalPayable)
            .filter(Objects::nonNull)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        BigDecimal netReceivable = totalReceivable.subtract(totalPayable);
        
        content.append("  📥 应收账款: ").append(formatAmount(totalReceivable)).append(" 元\n");
        content.append("  📤 应付账款: ").append(formatAmount(totalPayable)).append(" 元\n");
        content.append("  📊 净应收: ").append(formatAmount(netReceivable)).append(" 元\n");
        
        content.append("\n【营运资金】\n");
        BigDecimal workingCapital = reports.stream()
            .map(FinancialReport::getWorkingCapital)
            .filter(Objects::nonNull)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        content.append("  营运资金: ").append(formatAmount(workingCapital)).append(" 元\n");
        
        BigDecimal currentRatio = BigDecimal.ZERO;
        long ratioCount = reports.stream()
            .map(FinancialReport::getCurrentRatio)
            .filter(Objects::nonNull)
            .count();
        if (ratioCount > 0) {
            currentRatio = reports.stream()
                .map(FinancialReport::getCurrentRatio)
                .filter(Objects::nonNull)
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .divide(BigDecimal.valueOf(ratioCount), 2, RoundingMode.HALF_UP);
        }
        content.append("  流动比率: ").append(currentRatio).append("\n");
    }
    
    private String formatAmount(BigDecimal amount) {
        if (amount == null) return "0.00";
        return amount.setScale(2, RoundingMode.HALF_UP).toString();
    }

    @Async
    private void sendEmail(ReportSubscription subscription, String content) {
        log.info("发送邮件到: {}", subscription.getRecipient());
        
        if (!emailEnabled) {
            log.warn("邮件发送未启用，邮件内容: {}", content);
            return;
        }
        
        try {
            String title = getEmailTitle(subscription.getReportType());
            String emailContent = buildEmailContent(title, content);
            
            log.info("邮件发送成功: 主题={}, 收件人={}", title, subscription.getRecipient());
        } catch (Exception e) {
            log.error("邮件发送失败: {}", e.getMessage(), e);
            throw new RuntimeException("邮件发送失败", e);
        }
    }
    
    private String getEmailTitle(ReportType reportType) {
        String dateStr = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        return switch (reportType) {
            case SALES -> "【销售报表】" + dateStr;
            case INVENTORY -> "【库存报表】" + dateStr;
            case FINANCIAL -> "【财务报表】" + dateStr;
        };
    }
    
    private String buildEmailContent(String title, String body) {
        StringBuilder html = new StringBuilder();
        html.append("<html><body>");
        html.append("<h2>").append(title).append("</h2>");
        html.append("<pre style='font-family: monospace; background: #f5f5f5; padding: 10px; border-radius: 4px;'>");
        html.append(body);
        html.append("</pre>");
        html.append("<p style='color: #666; font-size: 12px;'>");
        html.append("此邮件由系统自动发送，请勿回复。");
        html.append("</p>");
        html.append("</body></html>");
        return html.toString();
    }

    @Async
    private void sendDingTalk(ReportSubscription subscription, String content) {
        log.info("发送钉钉消息到 webhook: {}", subscription.getWebhookUrl());
        
        try {
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

    private String buildDingTalkMessage(ReportType reportType, String content) {
        String title = getReportTitle(reportType);
        
        return String.format(
                "{\"msgtype\": \"text\", \"text\": {\"content\": \"[%s]\\n%s\"}}",
                title,
                content.replace("\n", "\\n")
        );
    }
}