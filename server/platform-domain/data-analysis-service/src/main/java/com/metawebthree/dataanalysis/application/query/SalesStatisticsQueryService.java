package com.metawebthree.dataanalysis.application.query;

import com.metawebthree.dataanalysis.application.dto.*;
import com.metawebthree.dataanalysis.domain.entity.SalesStatisticsDO;
import com.metawebthree.dataanalysis.infrastructure.persistence.mapper.SalesStatisticsMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class SalesStatisticsQueryService {

    private final SalesStatisticsMapper salesStatisticsMapper;

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