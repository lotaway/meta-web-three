package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.SalesHistoryDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface SalesHistoryMapper extends BaseMapper<SalesHistoryDO> {
}