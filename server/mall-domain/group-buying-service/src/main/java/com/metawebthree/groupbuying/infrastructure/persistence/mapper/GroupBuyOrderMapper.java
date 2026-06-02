package com.metawebthree.groupbuying.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.groupbuying.domain.model.GroupBuyOrderDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface GroupBuyOrderMapper extends BaseMapper<GroupBuyOrderDO> {
}