package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.BomVersionDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface BomVersionMapper extends BaseMapper<BomVersionDO> {
    
    BomVersionDO findByProductCode(@Param("productCode") String productCode);
}