package com.metawebthree.inventoryalert.infrastructure.persistence.mapper;

import com.metawebthree.inventoryalert.domain.model.InventoryAlertDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface InventoryAlertMapper {
    int insert(InventoryAlertDO record);
    int update(InventoryAlertDO record);
    int deleteById(Long id);
    InventoryAlertDO selectById(Long id);
    List<InventoryAlertDO> selectByProductId(Long productId);
    List<InventoryAlertDO> selectByWarehouseId(Long warehouseId);
    List<InventoryAlertDO> selectByAlertLevel(Integer alertLevel);
    List<InventoryAlertDO> selectByStatus(Integer status);
    List<InventoryAlertDO> selectAll();
    int updateStatus(@Param("id") Long id, @Param("status") Integer status);
    
    /**
     * Get count of unresolved (pending) alerts
     * @return count of alerts where alert_status = 0 (pending)
     */
    @Select("SELECT COUNT(*) FROM inventory_alert WHERE alert_status = 0")
    Long countPendingAlerts();
    
    /**
     * Get alert statistics by status
     * @return count grouped by alert_status
     */
    @Select("SELECT alert_status, COUNT(*) as count FROM inventory_alert GROUP BY alert_status")
    List<Object> selectAlertStatistics();
}