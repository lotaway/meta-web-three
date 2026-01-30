package com.metawebthree.user.application;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.metawebthree.user.domain.model.UserRoleDO;
import com.metawebthree.user.infrastructure.persistence.mapper.UserRoleMapper;

import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserRoleService extends ServiceImpl<UserRoleMapper, UserRoleDO> {

    private final UserRoleMapper userRoleMapper;

    public UserRoleService(UserRoleMapper userTypeMapper) {
        this.userRoleMapper = userTypeMapper;
    }

    public List<UserRoleDO> getList() {
        // MPJLambdaWrapper<UserPojo> wrapper = new MPJLambdaWrapper<>();
        // wrapper.select(UserPojo::getId);
        return userRoleMapper.selectList(null);
    }
}