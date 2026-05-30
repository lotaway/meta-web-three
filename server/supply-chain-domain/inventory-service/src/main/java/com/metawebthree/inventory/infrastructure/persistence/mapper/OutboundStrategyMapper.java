package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.OutboundStrategyDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface OutboundStrategyMapper extends BaseMapper<OutboundStrategyDO> {

    @Select("SELECT * FROM outbound_strategy WHERE warehouse_id = #{warehouseId} AND is_active = 1 ORDER BY priority ASC")
    List<OutboundStrategyDO> selectActiveByWarehouse(@Param("warehouseId") Long warehouseId);

    @Select("SELECT * FROM outbound_strategy WHERE (warehouse_id IS NULL OR warehouse_id = #{warehouseId}) AND is_active = 1 ORDER BY priority ASC")
    List<OutboundStrategyDO> selectActiveStrategies(@Param("warehouseId") Long warehouseId);
}