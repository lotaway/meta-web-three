package com.metawebthree.rma.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaOrderDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface RmaOrderMapper extends BaseMapper<RmaOrderDO> {
}
