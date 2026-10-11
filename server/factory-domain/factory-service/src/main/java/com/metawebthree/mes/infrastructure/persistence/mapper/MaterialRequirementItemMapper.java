package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialRequirementItemDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface MaterialRequirementItemMapper extends BaseMapper<MaterialRequirementItemDO> {
    
    List<MaterialRequirementItemDO> findByRequirementId(@Param("requirementId") Long requirementId);
    
    void deleteByRequirementId(@Param("requirementId") Long requirementId);
}