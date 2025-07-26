package com.metawebthree.user;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.github.yulichang.wrapper.MPJLambdaWrapper;

public interface IUserService extends IService<UserDO> {
    IPage<UserDO> getUserList(int pageNum);

    IPage<UserDO> getUserList(int pageNum, int pageSize);

    IPage<UserDO> getUserList(Integer pageNum, UserDO userPojo, Integer pageSize);

    IPage<UserDO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper);

    IPage<UserDO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper, Integer pageSize);

    int createUser(String email, String password) throws Exception;

    int createUser(String email, String password, Short typeId) throws Exception;

    default int updateUser(Integer id) {
        UserDO user = new UserDO();
        user.setId(id);
//        userMapper.updateUserInfo(user);
        return 0;
    }

    boolean deleteUser(Integer id);

    boolean deleteUsers(Integer[] ids);
}
