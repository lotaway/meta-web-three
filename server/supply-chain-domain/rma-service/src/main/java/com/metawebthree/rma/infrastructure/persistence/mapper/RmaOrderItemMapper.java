package com.metawebthree.rma.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaOrderItemDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface RmaOrderItemMapper extends BaseMapper<RmaOrderItemDO> {
}
