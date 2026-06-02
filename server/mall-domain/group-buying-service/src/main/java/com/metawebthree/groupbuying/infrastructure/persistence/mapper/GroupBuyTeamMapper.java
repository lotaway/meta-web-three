package com.metawebthree.groupbuying.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.groupbuying.domain.model.GroupBuyTeamDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface GroupBuyTeamMapper extends BaseMapper<GroupBuyTeamDO> {
}