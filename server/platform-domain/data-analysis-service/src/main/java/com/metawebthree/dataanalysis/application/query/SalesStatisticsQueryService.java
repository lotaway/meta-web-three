package com.metawebthree.dataanalysis.application.query;

import com.metawebthree.dataanalysis.application.dto.*;
import com.metawebthree.dataanalysis.domain.entity.SalesStatisticsDO;
import com.metawebthree.dataanalysis.domain.entity.HourlySalesDO;
import com.metawebthree.dataanalysis.domain.entity.ProductSalesDO;
import com.metawebthree.dataanalysis.infrastructure.client.InventoryAlertClient;
import com.metawebthree.dataanalysis.infrastructure.client.OrderClient;
import com.metawebthree.dataanalysis.infrastructure.client.PaymentClient;
import com.metawebthree.dataanalysis.infrastructure.persistence.mapper.SalesStatisticsMapper;
import com.metawebthree.dataanalysis.infrastructure.persistence.mapper.HourlySalesMapper;
import com.metawebthree.dataanalysis.infrastructure.persistence.mapper.ProductSalesMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class SalesStatisticsQueryService {

    private final SalesStatisticsMapper salesStatisticsMapper;
    private final HourlySalesMapper hourlySalesMapper;
    private final ProductSalesMapper productSalesMapper;
    private final OrderClient orderClient;
    private final InventoryAlertClient inventoryAlertClient;
    private final PaymentClient paymentClient;

    public SalesTrendDTO getSalesTrend(String startDate, String endDate) {
        List<SalesStatisticsDO> records = salesStatisticsMapper.selectByDateRange(startDate, endDate);
        
        List<SalesStatisticsDTO> trendList = records.stream()
            .map(this::toDTO)
            .collect(Collectors.toList());
        
        SalesTrendDTO result = new SalesTrendDTO();
        result.setTrendList(trendList);
        result.setTotalOrders(records.stream().mapToLong(r -> r.getOrderCount() != null ? r.getOrderCount() : 0L).sum());
        result.setTotalAmount(records.stream().mapToLong(r -> r.getTotalAmount() != null ? r.getTotalAmount() : 0L).sum());
        result.setTotalNewUsers(records.stream().mapToLong(r -> r.getNewUserCount() != null ? r.getNewUserCount() : 0L).sum());
        
        return result;
    }

    public SalesStatisticsDTO getDailySales(String date) {
        SalesStatisticsDO record = salesStatisticsMapper.selectByDate(date);
        return record != null ? toDTO(record) : null;
    }

    public List<CategorySalesDTO> getCategorySales(String startDate, String endDate) {
        List<SalesStatisticsDO> records = salesStatisticsMapper.selectByDateRange(startDate, endDate);
        List<CategorySalesDTO> result = new ArrayList<>();
        
        for (SalesStatisticsDO record : records) {
            CategorySalesDTO dto = new CategorySalesDTO();
            dto.setCategory("ALL");
            dto.setSalesAmount(record.getTotalAmount());
            dto.setOrderCount(record.getOrderCount());
            dto.setProportion(100.0);
            result.add(dto);
        }
        
        return result;
    }

    public RealTimeDashboardDTO getRealTimeDashboard() {
        String today = LocalDate.now().format(DateTimeFormatter.ISO_LOCAL_DATE);
        SalesStatisticsDO todayRecord = salesStatisticsMapper.selectByDate(today);
        
        // Get last week same day for comparison
        String lastWeekDate = LocalDate.now().minusWeeks(1).format(DateTimeFormatter.ISO_LOCAL_DATE);
        SalesStatisticsDO lastWeekRecord = salesStatisticsMapper.selectByDate(lastWeekDate);
        
        // Get last month same day for comparison
        String lastMonthDate = LocalDate.now().minusMonths(1).format(DateTimeFormatter.ISO_LOCAL_DATE);
        SalesStatisticsDO lastMonthRecord = salesStatisticsMapper.selectByDate(lastMonthDate);
        
        // Query hot products from database
        List<RealTimeDashboardDTO.HotProductDTO> hotProducts = queryHotProducts(today);
        
        // Query sales by hour from database
        List<RealTimeDashboardDTO.SalesByHourDTO> salesByHour = querySalesByHour(today);
        
        // Query order status distribution (would need order-service RPC call)
        Map<String, Integer> orderStatusDistribution = queryOrderStatusDistribution();
        
        // Query category sales distribution from product sales data
        Map<String, BigDecimal> categorySalesDistribution = queryCategorySalesDistribution(today);
        
        // Get real-time metrics from database or other services
        Long pendingOrders = queryPendingOrdersCount();
        Long lowStockAlerts = queryLowStockAlertsCount();
        Long pendingPayments = queryPendingPaymentsCount();
        
        // Calculate growth rates
        BigDecimal weekOverWeekGrowth = calculateGrowthRate(todayRecord, lastWeekRecord);
        BigDecimal monthOverMonthGrowth = calculateGrowthRate(todayRecord, lastMonthRecord);
        
        // Build real-time dashboard data
        return RealTimeDashboardDTO.builder()
                .todaySales(todayRecord != null && todayRecord.getTotalAmount() != null ? 
                        BigDecimal.valueOf(todayRecord.getTotalAmount()) : BigDecimal.ZERO)
                .todayOrders(todayRecord != null && todayRecord.getOrderCount() != null ? 
                        BigDecimal.valueOf(todayRecord.getOrderCount()) : BigDecimal.ZERO)
                .todayVisitors(todayRecord != null && todayRecord.getActiveUserCount() != null ? 
                        BigDecimal.valueOf(todayRecord.getActiveUserCount()) : BigDecimal.ZERO)
                .conversionRate(calculateConversionRate(todayRecord))
                .todayProfit(todayRecord != null && todayRecord.getOrderAmount() != null ? 
                        BigDecimal.valueOf(todayRecord.getOrderAmount()).multiply(BigDecimal.valueOf(0.15)) : BigDecimal.ZERO)
                .pendingOrders(pendingOrders != null ? pendingOrders.intValue() : 0)
                .lowStockAlerts(lowStockAlerts != null ? lowStockAlerts.intValue() : 0)
                .pendingPayments(pendingPayments != null ? pendingPayments.intValue() : 0)
                .hotProducts(hotProducts)
                .salesByHour(salesByHour)
                .orderStatusDistribution(orderStatusDistribution)
                .categorySalesDistribution(categorySalesDistribution)
                .weekOverWeekGrowth(weekOverWeekGrowth)
                .monthOverMonthGrowth(monthOverMonthGrowth)
                .build();
    }

    private BigDecimal calculateConversionRate(SalesStatisticsDO record) {
        if (record == null || record.getActiveUserCount() == null || record.getActiveUserCount() == 0) {
            return BigDecimal.ZERO;
        }
        if (record.getOrderCount() == null) {
            return BigDecimal.ZERO;
        }
        return BigDecimal.valueOf(record.getOrderCount() * 100.0 / record.getActiveUserCount())
                .setScale(2, RoundingMode.HALF_UP);
    }

    private BigDecimal calculateGrowthRate(SalesStatisticsDO current, SalesStatisticsDO previous) {
        if (current == null || current.getTotalAmount() == null || current.getTotalAmount() == 0) {
            return BigDecimal.ZERO;
        }
        if (previous == null || previous.getTotalAmount() == null || previous.getTotalAmount() == 0) {
            return BigDecimal.ZERO;
        }
        long currentAmount = current.getTotalAmount();
        long previousAmount = previous.getTotalAmount();
        return BigDecimal.valueOf((currentAmount - previousAmount) * 100.0 / previousAmount)
                .setScale(2, RoundingMode.HALF_UP);
    }

    /**
     * Query hot products from product_sales table
     */
    private List<RealTimeDashboardDTO.HotProductDTO> queryHotProducts(String date) {
        try {
            List<ProductSalesDO> products = productSalesMapper.selectTopProducts(date, 5);
            if (products != null && !products.isEmpty()) {
                return products.stream()
                    .map(p -> RealTimeDashboardDTO.HotProductDTO.builder()
                        .productId(String.valueOf(p.getProductId()))
                        .productName(p.getProductName())
                        .salesCount(p.getSalesCount() != null ? p.getSalesCount().intValue() : 0)
                        .salesAmount(p.getSalesAmount() != null ? 
                            BigDecimal.valueOf(p.getSalesAmount()) : BigDecimal.ZERO)
                        .build())
                    .collect(Collectors.toList());
            }
        } catch (Exception e) {
            log.warn("Failed to query hot products from database: {}", e.getMessage());
        }
        // Fallback: return empty list if no data
        return new ArrayList<>();
    }

    /**
     * Query sales by hour from hourly_sales table
     */
    private List<RealTimeDashboardDTO.SalesByHourDTO> querySalesByHour(String date) {
        try {
            List<HourlySalesDO> hourlyData = hourlySalesMapper.selectByDate(date);
            if (hourlyData != null && !hourlyData.isEmpty()) {
                return hourlyData.stream()
                    .map(h -> RealTimeDashboardDTO.SalesByHourDTO.builder()
                        .hour(h.getHour())
                        .sales(h.getTotalAmount() != null ? 
                            BigDecimal.valueOf(h.getTotalAmount()) : BigDecimal.ZERO)
                        .orders(h.getOrderCount() != null ? h.getOrderCount().intValue() : 0)
                        .build())
                    .collect(Collectors.toList());
            }
        } catch (Exception e) {
            log.warn("Failed to query sales by hour from database: {}", e.getMessage());
        }
        // Fallback: return empty list if no data
        return new ArrayList<>();
    }

    /**
     * Query order status distribution
     * Calls order-service via REST API
     */
    private Map<String, Integer> queryOrderStatusDistribution() {
        try {
            Map<String, Long> distribution = orderClient.getOrderStatusDistribution();
            // Convert Long values to Integer for the DTO
            Map<String, Integer> result = new HashMap<>();
            distribution.forEach((k, v) -> result.put(k, v != null ? v.intValue() : 0));
            return result;
        } catch (Exception e) {
            log.warn("Failed to query order status distribution: {}", e.getMessage());
            return new HashMap<>();
        }
    }

    /**
     * Query category sales distribution from product sales data
     */
    private Map<String, BigDecimal> queryCategorySalesDistribution(String date) {
        try {
            List<ProductSalesDO> products = productSalesMapper.selectByDate(date);
            if (products != null && !products.isEmpty()) {
                return products.stream()
                    .collect(Collectors.groupingBy(
                        p -> p.getCategory() != null ? p.getCategory() : "Unknown",
                        Collectors.reducing(
                            BigDecimal.ZERO,
                            p -> p.getSalesAmount() != null ? 
                                BigDecimal.valueOf(p.getSalesAmount()) : BigDecimal.ZERO,
                            BigDecimal::add
                        )
                    ));
            }
        } catch (Exception e) {
            log.warn("Failed to query category sales from database: {}", e.getMessage());
        }
        return new HashMap<>();
    }

    /**
     * Query pending orders count from order-service
     * Calls order-service via REST API
     */
    private Long queryPendingOrdersCount() {
        try {
            return orderClient.getPendingOrdersCount();
        } catch (Exception e) {
            log.warn("Failed to query pending orders count: {}", e.getMessage());
            return 0L;
        }
    }

    /**
     * Query low stock alerts count from inventory-alert-service
     * Calls inventory-alert-service via REST API
     */
    private Long queryLowStockAlertsCount() {
        try {
            return inventoryAlertClient.getLowStockAlertsCount();
        } catch (Exception e) {
            log.warn("Failed to query low stock alerts count: {}", e.getMessage());
            return 0L;
        }
    }

    /**
     * Query pending payments count from order-service
     * Pending payments are tracked in order status
     */
    private Long queryPendingPaymentsCount() {
        try {
            return orderClient.getPendingPaymentsCount();
        } catch (Exception e) {
            log.warn("Failed to query pending payments count: {}", e.getMessage());
            return 0L;
        }
    }

    private SalesStatisticsDTO toDTO(SalesStatisticsDO record) {
        SalesStatisticsDTO dto = new SalesStatisticsDTO();
        dto.setDate(record.getDate());
        dto.setOrderCount(record.getOrderCount());
        dto.setProductCount(record.getProductCount());
        dto.setTotalAmount(record.getTotalAmount());
        dto.setOrderAmount(record.getOrderAmount());
        dto.setRefundAmount(record.getRefundAmount());
        dto.setNewUserCount(record.getNewUserCount());
        dto.setActiveUserCount(record.getActiveUserCount());
        return dto;
    }
}