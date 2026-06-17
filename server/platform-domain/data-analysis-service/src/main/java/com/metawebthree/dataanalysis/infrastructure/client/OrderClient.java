package com.metawebthree.dataanalysis.infrastructure.client;

import com.metawebthree.common.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Component
public class OrderClient {

    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    /**
     * Get order status distribution counts
     * @return Map of status -> count
     */
    public Map<String, Long> getOrderStatusDistribution() {
        try {
            GetOrderStatusDistributionResponse response = orderService.getOrderStatusDistribution(
                    GetOrderStatusDistributionRequest.getDefaultInstance()
            );
            return new HashMap<>(response.getDistributionMap());
        } catch (Exception e) {
            log.error("Failed to get order status distribution via Dubbo", e);
            return new HashMap<>();
        }
    }

    /**
     * Get pending orders count
     * @return count of pending orders
     */
    public Long getPendingOrdersCount() {
        try {
            GetPendingOrdersCountResponse response = orderService.getPendingOrdersCount(
                    GetPendingOrdersCountRequest.getDefaultInstance()
            );
            return response.getCount();
        } catch (Exception e) {
            log.error("Failed to get pending orders count via Dubbo", e);
            return 0L;
        }
    }

    /**
     * Get pending payments count (orders awaiting payment)
     * @return count of orders waiting for payment
     */
    public Long getPendingPaymentsCount() {
        try {
            GetPendingPaymentsCountResponse response = orderService.getPendingPaymentsCount(
                    GetPendingPaymentsCountRequest.getDefaultInstance()
            );
            return response.getCount();
        } catch (Exception e) {
            log.error("Failed to get pending payments count via Dubbo", e);
            return 0L;
        }
    }

    /**
     * Get hot products based on sales
     * @param limit number of products to return
     * @return list of hot product info from proto
     */
    public List<HotProductInfo> getHotProducts(int limit) {
        try {
            GetHotProductsResponse response = orderService.getHotProducts(
                    GetHotProductsRequest.newBuilder().setLimit(limit).build()
            );
            return response.getProductsList();
        } catch (Exception e) {
            log.error("Failed to get hot products via Dubbo", e);
            return List.of();
        }
    }

    /**
     * Get sales by hour for today
     * @return list of hourly sales data from proto
     */
    public List<SalesByHourInfo> getSalesByHourToday() {
        try {
            GetSalesByHourTodayResponse response = orderService.getSalesByHourToday(
                    GetSalesByHourTodayRequest.getDefaultInstance()
            );
            return response.getHourlyDataList();
        } catch (Exception e) {
            log.error("Failed to get sales by hour today via Dubbo", e);
            return List.of();
        }
    }
}