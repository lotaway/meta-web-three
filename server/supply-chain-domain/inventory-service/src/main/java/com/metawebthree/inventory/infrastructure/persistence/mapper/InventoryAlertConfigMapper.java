package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryAlertConfigDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface InventoryAlertConfigMapper extends BaseMapper<InventoryAlertConfigDO> {
}