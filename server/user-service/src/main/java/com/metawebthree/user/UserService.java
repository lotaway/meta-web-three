package com.metawebthree.user;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.common.utils.UserRole;
import com.metawebthree.user.DO.UserDO;
import com.metawebthree.user.DTO.UserDTO;

public interface UserService extends IService<UserDO> {
    IPage<UserDTO> getUserList(int pageNum);

    IPage<UserDTO> getUserList(int pageNum, int pageSize);

    IPage<UserDTO> getUserList(Integer pageNum, UserDTO userDTO, Integer pageSize);

    IPage<UserDTO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper);

    IPage<UserDTO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper, Integer pageSize);

    Long createUser(String email, String password) throws Exception;

    Long createUser(String email, String password, UserRole userRoleId) throws Exception;

    default int updateUser(Long id) {
        UserDO user = new UserDO();
        user.setId(id);
        // userMapper.updateUser(user);
        return 0;
    }

    boolean deleteUser(Long id);

    boolean deleteUsers(Long[] ids);
}
