package com.metawebthree.server.mapper;

import com.metawebthree.server.pojo.User;
import org.apache.ibatis.annotations.*;

import java.util.ArrayList;

@Mapper
public interface UserMapper {
    @Select("select * from User limit #{pageSize},#{offset}")
    public ArrayList<User> getUserList(int pageSize, int offset);

    @Options(keyProperty = "id", useGeneratedKeys = true)
    @Insert("insert into User(email,password,typeId) values(#{email},#{password},#{typeId})")
    public int createUser(User user);

    @Update("update User set email=#{email},typeId=#{typeId} where id=#{id}")
    public int updateUserInfo(User user);

    @Update("update User set password=#{password} where id=#{id}")
    public int updateUserPassword(Integer id, String password);

    @Delete("delete from User where id=#{id}")
    public int deleteUser(Integer id);
}
