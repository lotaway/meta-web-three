package com.metawebthree.inventoryalert.infrastructure.persistence.mapper;

import com.metawebthree.inventoryalert.domain.model.InventoryAlertDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
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
}