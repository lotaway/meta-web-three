package com.metawebthree.digitaltwin.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.WarehouseDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface WarehouseMapper extends BaseMapper<WarehouseDO> {
}