package com.metawebthree.reporting.domain.service;

import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SalesReportDomainServiceTest {

    @Mock
    private SalesReportRepository repository;

    private SalesReportDomainService service;

    @BeforeEach
    void setUp() {
        service = new SalesReportDomainService(repository);
    }

    @Test
    void generateDailyReport_withData_shouldCalculateCorrectly() {
        LocalDateTime now = LocalDateTime.of(2026, 7, 20, 14, 30);
        LocalDateTime dayStart = now.toLocalDate().atStartOfDay();
        LocalDateTime dayEnd = now.toLocalDate().atTime(23, 59, 59);

        SalesReport existing1 = new SalesReport();
        existing1.setTotalSalesAmount(new BigDecimal("1000"));
        existing1.setTotalOrderCount(10);
        existing1.setCategoryBreakdown("Electronics");
        existing1.setChannelBreakdown("Online");

        SalesReport existing2 = new SalesReport();
        existing2.setTotalSalesAmount(new BigDecimal("2000"));
        existing2.setTotalOrderCount(20);
        existing2.setCategoryBreakdown("Clothing");
        existing2.setChannelBreakdown("Offline");

        when(repository.findByDateRange(dayStart, dayEnd)).thenReturn(List.of(existing1, existing2));

        SalesReport result = service.generateDailyReport(now);

        assertEquals("SALES-20260720", result.getReportNo());
        assertEquals(SalesReport.ReportType.DAILY, result.getType());
        assertEquals(now, result.getReportDate());
        assertEquals(dayStart, result.getStartDate());
        assertEquals(dayEnd, result.getEndDate());
        assertEquals(new BigDecimal("3000"), result.getTotalSalesAmount());
        assertEquals(30, result.getTotalOrderCount());
        assertEquals(new BigDecimal("100.00"), result.getAverageOrderAmount());
        assertEquals(new BigDecimal("750.00"), result.getGrossProfit());
        assertNotNull(result.getProfitMargin());
        assertNotNull(result.getCategoryBreakdown());
        assertNotNull(result.getChannelBreakdown());
    }

    @Test
    void generateDailyReport_withEmptyData_shouldUseDefaults() {
        LocalDateTime now = LocalDateTime.of(2026, 7, 20, 14, 30);
        LocalDateTime dayStart = now.toLocalDate().atStartOfDay();
        LocalDateTime dayEnd = now.toLocalDate().atTime(23, 59, 59);

        when(repository.findByDateRange(dayStart, dayEnd)).thenReturn(List.of());

        SalesReport result = service.generateDailyReport(now);

        assertEquals("SALES-20260720", result.getReportNo());
        assertEquals(SalesReport.ReportType.DAILY, result.getType());
        assertEquals(BigDecimal.ZERO, result.getTotalSalesAmount());
        assertEquals(0, result.getTotalOrderCount());
        assertEquals(BigDecimal.ZERO, result.getGrossProfit());
        assertEquals(BigDecimal.ZERO, result.getProfitMargin());
        assertEquals("{}", result.getCategoryBreakdown());
        assertEquals("{}", result.getChannelBreakdown());
    }

    @Test
    void generateMonthlyReport_withData_shouldCalculateCorrectly() {
        int year = 2026, month = 7;
        LocalDateTime startDate = LocalDateTime.of(year, month, 1, 0, 0);
        LocalDateTime endDate = startDate.plusMonths(1).minusSeconds(1);

        SalesReport existing1 = new SalesReport();
        existing1.setTotalSalesAmount(new BigDecimal("5000"));
        existing1.setTotalOrderCount(50);
        existing1.setCategoryBreakdown("Electronics");
        existing1.setProductRanking("Product A");
        existing1.setChannelBreakdown("Online");

        SalesReport existing2 = new SalesReport();
        existing2.setTotalSalesAmount(new BigDecimal("3000"));
        existing2.setTotalOrderCount(30);
        existing2.setCategoryBreakdown("Clothing");
        existing2.setProductRanking("Product B");
        existing2.setChannelBreakdown("Offline");

        when(repository.findByDateRange(startDate, endDate)).thenReturn(List.of(existing1, existing2));

        SalesReport result = service.generateMonthlyReport(year, month);

        assertEquals("SALES-202607", result.getReportNo());
        assertEquals(SalesReport.ReportType.MONTHLY, result.getType());
        assertEquals(new BigDecimal("8000"), result.getTotalSalesAmount());
        assertEquals(80, result.getTotalOrderCount());
        assertEquals(new BigDecimal("100.00"), result.getAverageOrderAmount());
        assertEquals(new BigDecimal("2240.00"), result.getGrossProfit());
        assertNotNull(result.getCategoryBreakdown());
        assertNotNull(result.getProductRanking());
        assertNotNull(result.getChannelBreakdown());
    }

    @Test
    void generateMonthlyReport_withEmptyData_shouldUseDefaults() {
        int year = 2026, month = 7;
        LocalDateTime startDate = LocalDateTime.of(year, month, 1, 0, 0);
        LocalDateTime endDate = startDate.plusMonths(1).minusSeconds(1);

        when(repository.findByDateRange(startDate, endDate)).thenReturn(List.of());

        SalesReport result = service.generateMonthlyReport(year, month);

        assertEquals("SALES-202607", result.getReportNo());
        assertEquals(SalesReport.ReportType.MONTHLY, result.getType());
        assertEquals(BigDecimal.ZERO, result.getTotalSalesAmount());
        assertEquals(0, result.getTotalOrderCount());
        assertEquals(new BigDecimal("500.00"), result.getAverageOrderAmount());
        assertEquals(BigDecimal.ZERO, result.getGrossProfit());
        assertEquals("{}", result.getCategoryBreakdown());
        assertEquals("[]", result.getProductRanking());
        assertEquals("{}", result.getChannelBreakdown());
    }
}
