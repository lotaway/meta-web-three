package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.DemandForecast;
import java.util.List;
import java.util.Optional;

public interface DemandForecastRepository {
    
    Optional<DemandForecast> findById(Long id);
    
    List<DemandForecast> findByStatus(String status);
    
    List<DemandForecast> findByWarehouseId(Long warehouseId);
    
    List<DemandForecast> findBySkuAndWarehouse(String skuCode, Long warehouseId);
    
    DemandForecast save(DemandForecast demandForecast);
    
    void delete(DemandForecast demandForecast);
}