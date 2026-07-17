package com.metawebthree.payment.interfaces.rpc;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import com.metawebthree.payment.infrastructure.persistence.mapper.ExchangeOrderRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Component;

import java.sql.Timestamp;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Dubbo RPC service implementation for payment statistics
 * Exposes payment statistics to other microservices
 */
@Slf4j
@DubboService
@Component
@RequiredArgsConstructor
public class PaymentRpcService implements PaymentService {

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    private final ExchangeOrderRepository exchangeOrderRepository;

    @Override
    public GetPaymentStatisticsResponse getPaymentStatistics(GetPaymentStatisticsRequest request) {
        log.info("Dubbo call: getPaymentStatistics");
        try {
            QueryWrapper<ExchangeOrder> totalQw = new QueryWrapper<>();
            long total = exchangeOrderRepository.selectCount(totalQw);

            long success = exchangeOrderRepository.selectCount(
                    new QueryWrapper<ExchangeOrder>().eq("status", ExchangeOrder.OrderStatus.COMPLETED));
            long failed = exchangeOrderRepository.selectCount(
                    new QueryWrapper<ExchangeOrder>().eq("status", ExchangeOrder.OrderStatus.FAILED));
            long pending = exchangeOrderRepository.selectCount(
                    new QueryWrapper<ExchangeOrder>().in("status",
                            ExchangeOrder.OrderStatus.PENDING,
                            ExchangeOrder.OrderStatus.PAID,
                            ExchangeOrder.OrderStatus.PROCESSING));

            PaymentStatistics statistics = PaymentStatistics.newBuilder()
                    .setTotalPayments(total)
                    .setSuccessPayments(success)
                    .setFailedPayments(failed)
                    .setPendingPayments(pending)
                    .build();
            return GetPaymentStatisticsResponse.newBuilder()
                    .setStatistics(statistics)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get payment statistics", e);
            throw new RuntimeException("Failed to get payment statistics", e);
        }
    }

    @Override
    public GetDailyPaymentStatsResponse getDailyPaymentStats(GetDailyPaymentStatsRequest request) {
        log.info("Dubbo call: getDailyPaymentStats for date: {}", request.getDate());
        try {
            LocalDate date = request.getDate().isEmpty()
                    ? LocalDate.now()
                    : LocalDate.parse(request.getDate(), DATE_FORMATTER);
            Timestamp start = Timestamp.valueOf(date.atStartOfDay());
            Timestamp end = Timestamp.valueOf(date.plusDays(1).atStartOfDay());

            List<ExchangeOrder> successOrders = exchangeOrderRepository.findByStatusAndCreatedAtBetween(
                    ExchangeOrder.OrderStatus.COMPLETED.name(), start, end);
            long successCount = successOrders.size();
            long successAmount = successOrders.stream()
                    .filter(o -> o.getFiatAmount() != null)
                    .mapToLong(o -> o.getFiatAmount().longValue())
                    .sum();

            List<ExchangeOrder> failedOrders = exchangeOrderRepository.findByStatusAndCreatedAtBetween(
                    ExchangeOrder.OrderStatus.FAILED.name(), start, end);
            long failedCount = failedOrders.size();

            String dateStr = date.format(DATE_FORMATTER);
            DailyPaymentStats stats = DailyPaymentStats.newBuilder()
                    .setDate(dateStr)
                    .setSuccessCount(successCount)
                    .setSuccessAmount(successAmount)
                    .setFailedCount(failedCount)
                    .build();
            return GetDailyPaymentStatsResponse.newBuilder()
                    .setStats(stats)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get daily payment stats", e);
            throw new RuntimeException("Failed to get daily payment stats", e);
        }
    }

    @Override
    public CompletableFuture<GetPaymentStatisticsResponse> getPaymentStatisticsAsync(GetPaymentStatisticsRequest request) {
        log.info("Dubbo async call: getPaymentStatistics");
        return CompletableFuture.completedFuture(getPaymentStatistics(request));
    }

    @Override
    public CompletableFuture<GetDailyPaymentStatsResponse> getDailyPaymentStatsAsync(GetDailyPaymentStatsRequest request) {
        log.info("Dubbo async call: getDailyPaymentStats for date: {}", request.getDate());
        return CompletableFuture.completedFuture(getDailyPaymentStats(request));
    }
}