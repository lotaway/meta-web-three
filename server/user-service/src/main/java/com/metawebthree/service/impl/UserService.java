package com.metawebthree.service.impl;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.mapper.UserMapper;
import com.metawebthree.pojo.AuthorPojo;
import com.metawebthree.pojo.UserPojo;
import com.metawebthree.service.IUserService;
import com.metawebthree.vo.PageConfig;
import org.apache.tomcat.util.security.MD5Encoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class UserService extends ServiceImpl<UserMapper, UserPojo> implements IUserService {

    private final PageConfig pageConfig;

    private final UserMapper userMapper;

//    private final OrderMapper orderMapper;

    public UserService(PageConfig pageConfig, UserMapper userMapper) {
        this.pageConfig = pageConfig;
        this.userMapper = userMapper;
    }

    public IPage<UserPojo> getUserList(int pageNum, UserPojo userPojo, AuthorPojo authorPojo) {
        //        QueryWrapper<UserPojo> userQueryWrapper = new QueryWrapper<>();
//        userQueryWrapper.select("id", "email", "authorId", "typeId").like("email", email).eq("user_type", user_type).eq("user_type", userType).and(wrapper -> wrapper.inSql("User.author_id", "select author_id from Author where real_name =" + realName));
        MPJLambdaWrapper<UserPojo> wrapper = new MPJLambdaWrapper<>();
        wrapper
                .select(UserPojo::getId, UserPojo::getEmail, UserPojo::getAuthorId, UserPojo::getTypeId)
                .select(AuthorPojo::getRealName)
                .innerJoin(AuthorPojo.class, AuthorPojo::getId, UserPojo::getAuthorId)
                .eq(AuthorPojo::getIsEnable, true)
                .like(AuthorPojo::getRealName, authorPojo.getRealName())
                .eq(UserPojo::getTypeId, userPojo.getTypeId())
                .eq(UserPojo::getEmail, userPojo.getEmail());
        return getUserList(pageNum, wrapper);
    }

    public IPage<UserPojo> getUserList() {
        return getUserList(1);
    }

    public IPage<UserPojo> getUserList(int pageNum) {
        return getUserList(pageNum, pageConfig.getPageSize());
    }

    public IPage<UserPojo> getUserList(int pageNum, int pageSize) {
        return getUserList(pageNum, (UserPojo) null, pageSize);
    }

    public IPage<UserPojo> getUserList(Integer pageNum, UserPojo userPojo, Integer pageSize) {
        Page<UserPojo> page = new Page<>(pageNum, pageSize);
        var wrapper = userPojo == null ? null : new MPJLambdaWrapper<>(userPojo);
        return userMapper.selectJoinPage(page, UserPojo.class, wrapper);
    }

    public IPage<UserPojo> getUserList(Integer pageNum, MPJLambdaWrapper<UserPojo> wrapper) {
        return getUserList(pageNum, wrapper, pageConfig.getPageSize());
    }

    public IPage<UserPojo> getUserList(Integer pageNum, MPJLambdaWrapper<UserPojo> wrapper, Integer pageSize) {
        Page<UserPojo> page = new Page<>(pageNum, pageSize);
        return userMapper.selectJoinPage(page, UserPojo.class, wrapper);
    }

    @Transactional
    public int createUser(String email, String password) {
        return createUser(email, password, (short) 0);
    }

    @Transactional
    public int createUser(String email, String password, Short typeId) {
        UserPojo userPojo = new UserPojo();
        userPojo.setEmail(email);
        userPojo.setPassword(MD5Encoder.encode(password.getBytes()));
        userPojo.setTypeId(typeId);
        int result = userMapper.createUser(userPojo);
//        userMapper.selectCount()
        return userPojo.getId();
    }

    public boolean deleteUser(Integer id) {
        return userMapper.deleteUser(id);
    }

    public boolean deleteUsers(Integer[] ids) {
        boolean result = true;
        userMapper.deleteUsers(ids);
        return result;
    }
}
