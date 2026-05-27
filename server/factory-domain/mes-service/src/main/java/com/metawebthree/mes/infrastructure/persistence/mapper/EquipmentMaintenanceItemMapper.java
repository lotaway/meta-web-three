package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentMaintenanceItemDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface EquipmentMaintenanceItemMapper extends BaseMapper<EquipmentMaintenanceItemDO> {
    
    @Select("SELECT * FROM mes_equipment_maintenance_item WHERE plan_id = #{planId} ORDER BY sort_order")
    List<EquipmentMaintenanceItemDO> selectByPlanId(@Param("planId") Long planId);
}