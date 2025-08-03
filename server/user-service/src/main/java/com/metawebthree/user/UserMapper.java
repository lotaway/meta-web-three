package com.metawebthree.user;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.user.DO.UserDO;

import org.apache.ibatis.annotations.*;

import java.util.ArrayList;

@Mapper
public interface UserMapper extends MPJBaseMapper<UserDO> {
    ArrayList<UserDO> getUserList(int pageSize, int offset, UserDO userPojo);

    @Options(keyProperty = "id", useGeneratedKeys = true)
    @Insert("insert into User(email,password,type_id) values(#{email},#{password},#{typeId})")
    int createUser(UserDO userDO);

    @Options(keyProperty = "id", useGeneratedKeys = true)
    @Insert("insert into User(email,password,type_id,wallet_address) values(#{email},#{password},#{typeId},#{walletAddress})")
    int createUserWithWallet(UserDO userDO);

    void updateUser(UserDO userDO);

    boolean deleteUser(Long id);

    void deleteUsers(Long[] ids);

    @Select("select * from User where email = #{email} and type_id = #{typeId}")
    UserDO findByEmailAndTypeId(@Param("email") String email, @Param("typeId") Short typeId);

    @Select("select * from Web3_User where wallet_address = #{walletAddress}")
    UserDO findByWalletAddress(@Param("walletAddress") String walletAddress);
}
