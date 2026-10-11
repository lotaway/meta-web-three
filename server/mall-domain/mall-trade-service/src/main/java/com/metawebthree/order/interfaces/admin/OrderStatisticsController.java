package com.metawebthree.order.interfaces.admin;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.order.application.AdminOrderQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * Admin order statistics controller
 * Provides order statistics for real-time dashboard
 */
@RestController
@RequestMapping("/api/admin/order/statistics")
@RequiredArgsConstructor
public class OrderStatisticsController {

    private final AdminOrderQueryService adminOrderQueryService;

    /**
     * Get order status distribution
     * @return map of status -> count
     */
    @GetMapping("/status-distribution")
    public ApiResponse<Map<String, Long>> getOrderStatusDistribution() {
        return ApiResponse.success(adminOrderQueryService.getOrderStatusDistribution());
    }

    /**
     * Get pending orders count
     * @return count of pending orders
     */
    @GetMapping("/pending-count")
    public ApiResponse<Long> getPendingOrdersCount() {
        return ApiResponse.success(adminOrderQueryService.getPendingOrdersCount());
    }
    
    /**
     * Get pending payments count (orders awaiting payment)
     * @return count of orders waiting for payment
     */
    @GetMapping("/pending-payments-count")
    public ApiResponse<Long> getPendingPaymentsCount() {
        return ApiResponse.success(adminOrderQueryService.getPendingPaymentsCount());
    }
}