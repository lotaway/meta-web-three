package com.metawebthree.user.infrastructure.persistence.mapper;
import com.metawebthree.user.domain.model.*;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.user.domain.model.Web3UserDO;

import org.apache.ibatis.annotations.*;

@Mapper
public interface Web3UserMapper extends MPJBaseMapper<Web3UserDO> {
}
