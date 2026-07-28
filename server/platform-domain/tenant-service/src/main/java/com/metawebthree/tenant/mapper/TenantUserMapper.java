package com.metawebthree.tenant.mapper;

import com.metawebthree.tenant.entity.TenantUser;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface TenantUserMapper extends BaseMapper<TenantUser> {
}
