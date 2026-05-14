package com.metawebthree.media.infrastructure.persistence.mapper;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.media.domain.model.UserStorageDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UserStorageMapper extends MPJBaseMapper<UserStorageDO> {
}
