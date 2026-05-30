package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.DemandForecast;
import java.util.List;

public interface DemandForecastDomainService {
    
    /**
     * 生成指定 SKU 的需求预测
     */
    DemandForecast generateForecast(String skuCode, Long warehouseId, Integer forecastDays, String method);
    
    /**
     * 为仓库生成所有 SKU 的需求预测
     */
    List<DemandForecast> generateForecastsForWarehouse(Long warehouseId, Integer forecastDays, String method);
    
    /**
     * 计算简单移动平均
     */
    Integer calculateSimpleMovingAverage(List<Integer> salesData, Integer windowSize);
    
    /**
     * 计算加权移动平均
     */
    Integer calculateWeightedMovingAverage(List<Integer> salesData, List<Integer> weights);
    
    /**
     * 获取预测列表
     */
    List<DemandForecast> getPendingForecasts();
    
    /**
     * 批准预测
     */
    DemandForecast approveForecast(Long forecastId);
    
    /**
     * 拒绝预测
     */
    DemandForecast rejectForecast(Long forecastId);
}