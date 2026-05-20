package com.metawebthree.user.infrastructure.persistence.mapper;
import com.metawebthree.user.domain.model.*;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.user.domain.model.UserRoleMappingDO;

import org.apache.ibatis.annotations.*;

@Mapper
public interface UserRoleMappingMapper extends MPJBaseMapper<UserRoleMappingDO> {
}
