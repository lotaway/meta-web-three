package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.metawebthree.inventory.infrastructure.persistence.dataobject.DemandForecastDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import java.util.List;

@Mapper
public interface DemandForecastMapper {
    
    @Select("SELECT * FROM demand_forecast WHERE id = #{id}")
    DemandForecastDO selectById(Long id);
    
    @Select("SELECT * FROM demand_forecast WHERE status = #{status}")
    List<DemandForecastDO> selectByStatus(String status);
    
    @Select("SELECT * FROM demand_forecast WHERE warehouse_id = #{warehouseId}")
    List<DemandForecastDO> selectByWarehouseId(Long warehouseId);
    
    @Select("SELECT * FROM demand_forecast WHERE sku_code = #{skuCode} AND warehouse_id = #{warehouseId}")
    List<DemandForecastDO> selectBySkuAndWarehouse(String skuCode, Long warehouseId);
    
    int insert(DemandForecastDO demandForecastDO);
    
    int update(DemandForecastDO demandForecastDO);
    
    int deleteById(Long id);
}