package com.metawebthree.user.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.user.domain.model.MemberLevelDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface MemberLevelMapper extends BaseMapper<MemberLevelDO> {
}
