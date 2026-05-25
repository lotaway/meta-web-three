package com.metawebthree.digitaltwin.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.InventoryAlertDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface InventoryAlertMapper extends BaseMapper<InventoryAlertDO> {
}