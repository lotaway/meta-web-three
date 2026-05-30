package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.DemandForecastDTO;
import java.util.List;

public interface DemandForecastApplicationService {
    
    /**
     * 生成单个 SKU 的需求预测
     */
    DemandForecastDTO generateForecast(String skuCode, Long warehouseId, Integer forecastDays, String method);
    
    /**
     * 为仓库生成所有 SKU 的需求预测
     */
    List<DemandForecastDTO> generateForecastsForWarehouse(Long warehouseId, Integer forecastDays, String method);
    
    /**
     * 获取待处理的预测列表
     */
    List<DemandForecastDTO> getPendingForecasts();
    
    /**
     * 批准预测
     */
    DemandForecastDTO approveForecast(Long forecastId);
    
    /**
     * 拒绝预测
     */
    DemandForecastDTO rejectForecast(Long forecastId);
    
    /**
     * 根据 ID 查询预测
     */
    DemandForecastDTO queryById(Long id);
    
    /**
     * 根据仓库查询预测
     */
    List<DemandForecastDTO> queryByWarehouse(Long warehouseId);
}