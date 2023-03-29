package com.metawebthree.server.service;

import com.metawebthree.server.mapper.UserMapper;
import com.metawebthree.server.pojo.User;
import org.apache.tomcat.util.security.MD5Encoder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;

@Service
public class UserService {

    @Autowired
    private UserMapper userMapper;

    public ArrayList<com.metawebthree.server.pojo.User> getUserList(int pageIndex, int pageSize) {
        return userMapper.getUserList(pageSize, pageIndex - 1);
    }

    public ArrayList<com.metawebthree.server.pojo.User> getUserList(int pageIndex) {
        return getUserList(pageIndex, 20);
    }

    public int createUser(String email, String password, Short typeId) {
        User user = new User();
        user.setEmail(email);
        user.setPassword(MD5Encoder.encode(password.getBytes()));
        user.setTypeId(typeId);
        int result = userMapper.createUser(user);
        Integer userId = user.getId();
        LocalDateTime localDateTime = LocalDateTime.now();
        System.out.print("userId:" + userId + ", date_time:" + localDateTime);
        return result;
    }

    /*public int updateUserInfo(Integer id) {
        User user = new User();
        user.setId(id);
        return userMapper.updateUserInfo(user);
    }*/

    public int deleteUser(Integer id) {
        return userMapper.deleteUser(id);
    }
}
