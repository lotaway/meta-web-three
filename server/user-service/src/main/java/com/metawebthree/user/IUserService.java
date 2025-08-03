package com.metawebthree.user;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.user.DO.UserDO;

public interface IUserService extends IService<UserDO> {
    IPage<UserDO> getUserList(int pageNum);

    IPage<UserDO> getUserList(int pageNum, int pageSize);

    IPage<UserDO> getUserList(Integer pageNum, UserDO userPojo, Integer pageSize);

    IPage<UserDO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper);

    IPage<UserDO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper, Integer pageSize);

    Long createUser(String email, String password) throws Exception;

    Long createUser(String email, String password, Short typeId) throws Exception;

    default int updateUser(Long id) {
        UserDO user = new UserDO();
        user.setId(id);
        // userMapper.updateUserInfo(user);
        return 0;
    }

    boolean deleteUser(Long id);

    boolean deleteUsers(Long[] ids);
}
