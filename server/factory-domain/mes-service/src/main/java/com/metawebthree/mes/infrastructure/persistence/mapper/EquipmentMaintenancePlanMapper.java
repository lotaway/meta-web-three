package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentMaintenancePlanDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface EquipmentMaintenancePlanMapper extends BaseMapper<EquipmentMaintenancePlanDO> {
    
    @Select("SELECT * FROM mes_equipment_maintenance_plan WHERE plan_code = #{planCode}")
    EquipmentMaintenancePlanDO selectByPlanCode(@Param("planCode") String planCode);
    
    @Select("SELECT * FROM mes_equipment_maintenance_plan WHERE equipment_type_code = #{equipmentTypeCode}")
    List<EquipmentMaintenancePlanDO> selectByEquipmentTypeCode(@Param("equipmentTypeCode") String equipmentTypeCode);
    
    @Select("SELECT * FROM mes_equipment_maintenance_plan WHERE is_active = #{isActive}")
    List<EquipmentMaintenancePlanDO> selectByIsActive(@Param("isActive") Boolean isActive);
}