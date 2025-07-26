package com.metawebthree.user;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.author.AuthorPojo;
import com.metawebthree.common.PageConfigVO;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class UserService extends ServiceImpl<UserMapper, UserDO> implements IUserService {

    private final PageConfigVO pageConfigVo;

    private final UserMapper userMapper;

//    private final OrderMapper orderMapper;

    public UserService(PageConfigVO pageConfigVo, UserMapper userMapper) {
        this.pageConfigVo = pageConfigVo;
        this.userMapper = userMapper;
    }

    public IPage<UserDO> getUserList(int pageNum, UserDO userPojo, AuthorPojo authorPojo) {
        //        QueryWrapper<UserPojo> userQueryWrapper = new QueryWrapper<>();
//        userQueryWrapper.select("id", "email", "authorId", "typeId").like("email", email).eq("user_type", user_type).eq("user_type", userType).and(wrapper -> wrapper.inSql("User.author_id", "select author_id from Author where real_name =" + realName));
        MPJLambdaWrapper<UserDO> wrapper = new MPJLambdaWrapper<>();
        wrapper
                .select(UserDO::getId, UserDO::getEmail, UserDO::getAuthorId, UserDO::getTypeId)
                .select(AuthorPojo::getRealName)
                .innerJoin(AuthorPojo.class, AuthorPojo::getId, UserDO::getAuthorId)
                .eq(AuthorPojo::getIsEnable, true)
                .like(AuthorPojo::getRealName, authorPojo.getRealName())
                .eq(UserDO::getTypeId, userPojo.getTypeId())
                .eq(UserDO::getEmail, userPojo.getEmail());
        return getUserList(pageNum, wrapper);
    }

    public IPage<UserDO> getUserList() {
        return getUserList(1);
    }

    public IPage<UserDO> getUserList(int pageNum) {
        return getUserList(pageNum, pageConfigVo.getPageSize());
    }

    public IPage<UserDO> getUserList(int pageNum, int pageSize) {
        return getUserList(pageNum, (UserDO) null, pageSize);
    }

    public IPage<UserDO> getUserList(Integer pageNum, UserDO userPojo, Integer pageSize) {
        Page<UserDO> page = new Page<>(pageNum, pageSize);
        var wrapper = userPojo == null ? null : new MPJLambdaWrapper<>(userPojo);
        return userMapper.selectJoinPage(page, UserDO.class, wrapper);
    }

    public IPage<UserDO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper) {
        return getUserList(pageNum, wrapper, pageConfigVo.getPageSize());
    }

    public IPage<UserDO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper, Integer pageSize) {
        Page<UserDO> page = new Page<>(pageNum, pageSize);
        return userMapper.selectJoinPage(page, UserDO.class, wrapper);
    }

    @Transactional
    public int createUser(String email, String password) throws NoSuchAlgorithmException {
        return createUser(email, password, (short) 0);
    }

    @Transactional
    public int createUser(String email, String password, Short typeId) throws NoSuchAlgorithmException {
        UserDO userDO = new UserDO();
        userDO.setEmail(email);
        userDO.setPassword(md5Encrypt(password));
        userDO.setTypeId(typeId);
        int result = userMapper.createUser(userDO);
        System.out.println("User created with ID: " + userDO.getId() + " and result: " + result);
//        userMapper.selectCount()
        return userDO.getId();
    }


    private String md5Encrypt(String password) throws NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("MD5");
        byte[] hash = md.digest(password.getBytes());
        StringBuilder sb = new StringBuilder();
        for (byte b : hash) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
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
