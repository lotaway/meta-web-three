package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkOrderCodeRuleDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface WorkOrderCodeRuleMapper extends BaseMapper<WorkOrderCodeRuleDO> {
    
    @Select("SELECT * FROM mes_work_order_code_rule WHERE workshop_id = #{workshopId} " +
            "AND work_order_type = #{workOrderType} AND is_active = true ORDER BY priority DESC LIMIT 1")
    WorkOrderCodeRuleDO findActiveByWorkshopAndType(@Param("workshopId") String workshopId, 
                                                     @Param("workOrderType") String workOrderType);
    
    @Select("SELECT * FROM mes_work_order_code_rule WHERE work_order_type = #{workOrderType} " +
            "AND is_active = true ORDER BY priority DESC LIMIT 1")
    WorkOrderCodeRuleDO findActiveByType(@Param("workOrderType") String workOrderType);
    
    @Select("SELECT * FROM mes_work_order_code_rule WHERE workshop_id = #{workshopId} " +
            "AND is_active = true ORDER BY priority DESC LIMIT 1")
    WorkOrderCodeRuleDO findActiveByWorkshop(@Param("workshopId") String workshopId);
    
    @Select("SELECT * FROM mes_work_order_code_rule WHERE is_active = true ORDER BY priority DESC")
    List<WorkOrderCodeRuleDO> findAllActive();
    
    @Select("SELECT * FROM mes_work_order_code_rule WHERE rule_code = #{ruleCode} AND is_active = true LIMIT 1")
    WorkOrderCodeRuleDO findActiveByRuleCode(@Param("ruleCode") String ruleCode);
}