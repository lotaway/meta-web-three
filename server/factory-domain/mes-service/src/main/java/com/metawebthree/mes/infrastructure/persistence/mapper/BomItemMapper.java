package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.BomItemDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface BomItemMapper extends BaseMapper<BomItemDO> {
    
    List<BomItemDO> findByBomId(@Param("bomId") Long bomId);
    
    void deleteByBomId(@Param("bomId") Long bomId);
}