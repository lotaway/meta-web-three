package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryBatchDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface InventoryBatchMapper extends BaseMapper<InventoryBatchDO> {

    @Select("SELECT * FROM inventory_batch WHERE sku_code = #{skuCode} AND warehouse_id = #{warehouseId} AND status = 'AVAILABLE' AND available_quantity > 0 ORDER BY inbound_date ASC, id ASC")
    List<InventoryBatchDO> selectAvailableBatchesFifo(@Param("skuCode") String skuCode, @Param("warehouseId") Long warehouseId);

    @Select("SELECT * FROM inventory_batch WHERE sku_code = #{skuCode} AND warehouse_id = #{warehouseId} AND status = 'AVAILABLE' AND available_quantity > 0 ORDER BY inbound_date DESC, id DESC")
    List<InventoryBatchDO> selectAvailableBatchesLifo(@Param("skuCode") String skuCode, @Param("warehouseId") Long warehouseId);

    @Select("SELECT * FROM inventory_batch WHERE sku_code = #{skuCode} AND warehouse_id = #{warehouseId} AND batch_no = #{batchNo}")
    List<InventoryBatchDO> selectByBatchNo(@Param("skuCode") String skuCode, @Param("warehouseId") Long warehouseId, @Param("batchNo") String batchNo);
}