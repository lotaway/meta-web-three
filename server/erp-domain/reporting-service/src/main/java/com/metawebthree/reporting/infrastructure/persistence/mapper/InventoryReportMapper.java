package com.metawebthree.reporting.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.InventoryReportDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface InventoryReportMapper extends BaseMapper<InventoryReportDO> {
}