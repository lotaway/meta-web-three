package com.metawebthree.user;

import com.github.yulichang.base.MPJBaseMapper;
import org.apache.ibatis.annotations.*;

import java.util.ArrayList;

@Mapper
public interface UserMapper extends MPJBaseMapper<UserDO> {
    ArrayList<UserDO> getUserList(int pageSize, int offset, UserDO userPojo);

    @Options(keyProperty = "id", useGeneratedKeys = true)
    @Insert("insert into User(email,password,type_id) values(#{email},#{password},#{typeId})")
    int createUser(UserDO userPojo);

    @Options(keyProperty = "id", useGeneratedKeys = true)
    @Insert("insert into User(email,password,type_id,wallet_address) values(#{email},#{password},#{typeId},#{walletAddress})")
    int createUserWithWallet(UserDO userPojo);

    void updateUser(UserDO userPojo);

    @Delete("delete from User where id=#{id}")
    boolean deleteUser(Integer id);

    void deleteUsers(Integer[] ids);

    @Select("select * from User where email = #{email} and type_id = #{typeId}")
    UserDO findByEmailAndTypeId(@Param("email") String email, @Param("typeId") Short typeId);

    @Select("select * from User where wallet_address = #{walletAddress}")
    UserDO findByWalletAddress(@Param("walletAddress") String walletAddress);
}
