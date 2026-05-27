package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialIssueConfigDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface MaterialIssueConfigMapper extends BaseMapper<MaterialIssueConfigDO> {
    
    @Select("SELECT * FROM mes_material_issue_config WHERE workshop_id = #{workshopId} " +
            "AND product_code = #{productCode} AND is_active = true ORDER BY priority DESC LIMIT 1")
    MaterialIssueConfigDO findActiveByWorkshopAndProduct(@Param("workshopId") String workshopId,
                                                          @Param("productCode") String productCode);
    
    @Select("SELECT * FROM mes_material_issue_config WHERE workshop_id = #{workshopId} " +
            "AND product_code IS NULL AND is_active = true ORDER BY priority DESC LIMIT 1")
    MaterialIssueConfigDO findActiveByWorkshopDefault(@Param("workshopId") String workshopId);
    
    @Select("SELECT * FROM mes_material_issue_config WHERE is_active = true ORDER BY priority DESC")
    List<MaterialIssueConfigDO> findAllActive();
    
    @Select("SELECT * FROM mes_material_issue_config WHERE config_code = #{configCode} AND is_active = true LIMIT 1")
    MaterialIssueConfigDO findActiveByConfigCode(@Param("configCode") String configCode);
    
    @Select("SELECT * FROM mes_material_issue_config WHERE issue_mode = #{issueMode} AND is_active = true")
    List<MaterialIssueConfigDO> findActiveByIssueMode(@Param("issueMode") String issueMode);
}