package com.metawebthree.user.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.user.domain.model.AdminRoleRelationDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface AdminRoleRelationMapper extends BaseMapper<AdminRoleRelationDO> {
}
