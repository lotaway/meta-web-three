package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessBomItemDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface ProcessBomItemMapper extends BaseMapper<ProcessBomItemDO> {
    
    List<ProcessBomItemDO> findByRouteId(@Param("routeId") String routeId);
    
    List<ProcessBomItemDO> findByProcessCode(@Param("routeId") String routeId, @Param("processCode") String processCode);
}