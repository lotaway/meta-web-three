package com.metawebthree.reporting.application.query;

import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SalesReportQueryServiceTest {

    @Mock
    private SalesReportRepository repository;

    private SalesReportQueryService service;

    @BeforeEach
    void setUp() {
        service = new SalesReportQueryService(repository);
    }

    @Test
    void generateDailyReport_shouldSaveReport() {
        service.generateDailyReport();

        verify(repository).save(any(SalesReport.class));
    }

    @Test
    void generateMonthlyReport_shouldSaveReport() {
        service.generateMonthlyReport(2026, 7);

        verify(repository).save(any(SalesReport.class));
    }

    @Test
    void getById_whenExists_shouldReturnReport() {
        SalesReport report = new SalesReport();
        report.setId(1L);
        when(repository.findById(1L)).thenReturn(Optional.of(report));

        Optional<SalesReport> result = service.getById(1L);

        assertTrue(result.isPresent());
        assertEquals(1L, result.get().getId());
    }

    @Test
    void getById_whenNotExists_shouldReturnEmpty() {
        when(repository.findById(99L)).thenReturn(Optional.empty());

        Optional<SalesReport> result = service.getById(99L);

        assertTrue(result.isEmpty());
    }

    @Test
    void listByType_shouldReturnFilteredList() {
        SalesReport daily = new SalesReport();
        daily.setType(SalesReport.ReportType.DAILY);
        when(repository.findByType(SalesReport.ReportType.DAILY)).thenReturn(List.of(daily));

        List<SalesReport> result = service.listByType("DAILY");

        assertEquals(1, result.size());
        assertEquals(SalesReport.ReportType.DAILY, result.get(0).getType());
    }

    @Test
    void listAll_shouldReturnAllReports() {
        SalesReport r1 = new SalesReport();
        SalesReport r2 = new SalesReport();
        when(repository.findAll()).thenReturn(List.of(r1, r2));

        List<SalesReport> result = service.listAll();

        assertEquals(2, result.size());
    }
}
