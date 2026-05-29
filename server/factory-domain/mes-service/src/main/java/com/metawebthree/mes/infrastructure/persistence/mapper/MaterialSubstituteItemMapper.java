package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialSubstituteItemDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface MaterialSubstituteItemMapper extends BaseMapper<MaterialSubstituteItemDO> {
    
    List<MaterialSubstituteItemDO> findByGroupId(@Param("groupId") Long groupId);
    
    void deleteByGroupId(@Param("groupId") Long groupId);
}