package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryRecordDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface InventoryRecordMapper extends BaseMapper<InventoryRecordDO> {
}