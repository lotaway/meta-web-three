package com.metawebthree.dataanalysis.interfaces.controller;

import com.metawebthree.dataanalysis.application.dto.*;
import com.metawebthree.dataanalysis.application.query.SalesStatisticsQueryService;
import com.metawebthree.dataanalysis.application.query.UserPortraitQueryService;
import com.metawebthree.dataanalysis.application.query.InventoryAnalysisQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/analysis")
@RequiredArgsConstructor
public class DataAnalysisController {

    private final SalesStatisticsQueryService salesStatisticsQueryService;
    private final UserPortraitQueryService userPortraitQueryService;
    private final InventoryAnalysisQueryService inventoryAnalysisQueryService;

    @GetMapping("/sales/trend")
    public SalesTrendDTO getSalesTrend(
            @RequestParam String startDate,
            @RequestParam String endDate) {
        return salesStatisticsQueryService.getSalesTrend(startDate, endDate);
    }

    @GetMapping("/sales/daily")
    public SalesStatisticsDTO getDailySales(@RequestParam String date) {
        return salesStatisticsQueryService.getDailySales(date);
    }

    @GetMapping("/sales/category")
    public List<CategorySalesDTO> getCategorySales(
            @RequestParam String startDate,
            @RequestParam String endDate) {
        return salesStatisticsQueryService.getCategorySales(startDate, endDate);
    }

    @GetMapping("/user/portrait")
    public UserPortraitDTO getUserPortrait(
            @RequestParam String startDate,
            @RequestParam String endDate) {
        return userPortraitQueryService.getUserPortrait(startDate, endDate);
    }

    @GetMapping("/user/profile/{userId}")
    public UserProfileDTO getUserProfile(@PathVariable Long userId) {
        return userPortraitQueryService.getUserProfile(userId);
    }

    @GetMapping("/inventory/overview")
    public InventoryOverviewDTO getInventoryOverview() {
        return inventoryAnalysisQueryService.getInventoryOverview();
    }

    @GetMapping("/inventory/product/{productId}")
    public InventoryAnalysisDTO getProductInventory(@PathVariable String productId) {
        return inventoryAnalysisQueryService.getProductInventory(productId);
    }

    @GetMapping("/inventory/low-stock")
    public List<InventoryAnalysisDTO> getLowStockProducts() {
        return inventoryAnalysisQueryService.getLowStockProducts();
    }

    @GetMapping("/inventory/overstock")
    public List<InventoryAnalysisDTO> getOverstockProducts() {
        return inventoryAnalysisQueryService.getOverstockProducts();
    }
}