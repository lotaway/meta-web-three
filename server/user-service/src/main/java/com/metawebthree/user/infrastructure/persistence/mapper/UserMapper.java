package com.metawebthree.user.infrastructure.persistence.mapper;
import com.metawebthree.user.domain.model.*;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.user.domain.model.UserDO;
import com.metawebthree.user.application.dto.UserDTO;

import org.apache.ibatis.annotations.*;

import java.util.ArrayList;

@Mapper
public interface UserMapper extends MPJBaseMapper<UserDO> {
    ArrayList<UserDO> getUserList(int pageSize, int offset, UserDO userPojo);

    @Options(keyProperty = "id", useGeneratedKeys = true)
    @Insert("insert into User(email,password,type_id) values(#{email},#{password},#{typeId})")
    int createUser(UserDO userDO);

    @Options(keyProperty = "id", useGeneratedKeys = true)

    void updateUser(UserDO userDO);

    boolean deleteUser(Long id);

    void deleteUsers(Long[] ids);

    @Select("select User.*, User_Role_Mapping.user_role_id, Web3_User.wallet_address from User, User_Role_Mapping, Web3_User where email = #{email} and User.id = User_Role_Mapping.user_id and User_Role_Mapping.user_role_id = #{userRoleId} and User.id = Web3_User.user_id group by User.id")
    UserDTO findByEmailAndTypeId(@Param("email") String email, @Param("userRoleId") Long userRoleId);

    @Select("select User.*, Web3_User.wallet_address from User, Web3_User where Web3_User.user_id = User.id and Web3_User.wallet_address = #{walletAddress} group by User.id")
    UserDTO findByWalletAddress(@Param("walletAddress") String walletAddress);
}
