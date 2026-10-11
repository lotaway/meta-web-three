package com.metawebthree.groupbuying.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.groupbuying.domain.model.GroupBuyActivityDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface GroupBuyActivityMapper extends BaseMapper<GroupBuyActivityDO> {
}