package com.metawebthree.warehouse.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.WarehouseDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface WarehouseMapper extends BaseMapper<WarehouseDO> {
}