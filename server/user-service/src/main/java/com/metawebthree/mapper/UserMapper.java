package com.metawebthree.mapper;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.pojo.UserPojo;
import org.apache.ibatis.annotations.*;

import java.util.ArrayList;

@Mapper
public interface UserMapper extends MPJBaseMapper<UserPojo> {
    ArrayList<UserPojo> getUserList(int pageSize, int offset, UserPojo userPojo);

    @Options(keyProperty = "id", useGeneratedKeys = true)
    @Insert("insert into User(email,password,type_id) values(#{email},#{password},#{typeId})")
    int createUser(UserPojo userPojo);

    void updateUser(UserPojo userPojo);

    @Delete("delete from User where id=#{id}")
    boolean deleteUser(Integer id);

    void deleteUsers(Integer[] ids);
}
