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

    void updateUser(UserDO userPojo);

    @Delete("delete from User where id=#{id}")
    boolean deleteUser(Integer id);

    void deleteUsers(Integer[] ids);
}
