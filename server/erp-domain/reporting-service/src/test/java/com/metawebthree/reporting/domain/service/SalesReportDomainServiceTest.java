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

    private List<SalesReport> twoDailyReports() {
        SalesReport r1 = new SalesReport();
        r1.setTotalSalesAmount(new BigDecimal("1000"));
        r1.setTotalOrderCount(10);
        r1.setCategoryBreakdown("Electronics");
        r1.setChannelBreakdown("Online");
        SalesReport r2 = new SalesReport();
        r2.setTotalSalesAmount(new BigDecimal("2000"));
        r2.setTotalOrderCount(20);
        r2.setCategoryBreakdown("Clothing");
        r2.setChannelBreakdown("Offline");
        return List.of(r1, r2);
    }

    private List<SalesReport> twoMonthlyReports() {
        SalesReport r1 = new SalesReport();
        r1.setTotalSalesAmount(new BigDecimal("5000"));
        r1.setTotalOrderCount(50);
        r1.setCategoryBreakdown("Electronics");
        r1.setProductRanking("Product A");
        r1.setChannelBreakdown("Online");
        SalesReport r2 = new SalesReport();
        r2.setTotalSalesAmount(new BigDecimal("3000"));
        r2.setTotalOrderCount(30);
        r2.setCategoryBreakdown("Clothing");
        r2.setProductRanking("Product B");
        r2.setChannelBreakdown("Offline");
        return List.of(r1, r2);
    }

    @Test
    void generateDailyReport_withData_shouldCalculateCorrectly() {
        LocalDateTime now = LocalDateTime.of(2026, 7, 20, 14, 30);
        LocalDateTime dayStart = now.toLocalDate().atStartOfDay();
        LocalDateTime dayEnd = now.toLocalDate().atTime(23, 59, 59);
        when(repository.findByDateRange(dayStart, dayEnd)).thenReturn(twoDailyReports());

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
        verify(repository).findByDateRange(dayStart, dayEnd);
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
        verify(repository).findByDateRange(dayStart, dayEnd);
    }

    @Test
    void generateMonthlyReport_withData_shouldCalculateCorrectly() {
        int year = 2026, month = 7;
        LocalDateTime startDate = LocalDateTime.of(year, month, 1, 0, 0);
        LocalDateTime endDate = startDate.plusMonths(1).minusSeconds(1);
        when(repository.findByDateRange(startDate, endDate)).thenReturn(twoMonthlyReports());

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
        verify(repository).findByDateRange(startDate, endDate);
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
        verify(repository).findByDateRange(startDate, endDate);
    }
}
