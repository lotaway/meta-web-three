package com.metawebthree.user.infrastructure.persistence.mapper;
import com.metawebthree.user.domain.model.*;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.user.domain.model.UserRoleDO;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UserRoleMapper extends MPJBaseMapper<UserRoleDO> {

}