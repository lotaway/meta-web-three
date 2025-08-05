package com.metawebthree.user.impl;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.author.AuthorDO;
import com.metawebthree.common.PageConfigVO;
import com.metawebthree.user.UserService;
import com.metawebthree.user.UserMapper;
import com.metawebthree.user.DO.UserDO;
import com.metawebthree.user.DO.Web3UserDO;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class UserServiceImpl extends ServiceImpl<UserMapper, UserDO> implements UserService {

    private final PageConfigVO pageConfigVo;

    private final UserMapper userMapper;

    public UserServiceImpl(PageConfigVO pageConfigVo, UserMapper userMapper) {
        this.pageConfigVo = pageConfigVo;
        this.userMapper = userMapper;
    }

    public IPage<UserDO> getUserList(int pageNum, UserDO userPojo, AuthorDO authorPojo) {
        // QueryWrapper<UserPojo> userQueryWrapper = new QueryWrapper<>();
        // userQueryWrapper.select("id", "email", "authorId", "typeId").like("email",
        // email).eq("user_type", user_type).eq("user_type", userType).and(wrapper ->
        // wrapper.inSql("User.author_id", "select author_id from Author where real_name
        // =" + realName));
        MPJLambdaWrapper<UserDO> wrapper = new MPJLambdaWrapper<>();
        wrapper.select(UserDO::getId, UserDO::getEmail, UserDO::getAuthorId, UserDO::getTypeId)
                .select(AuthorDO::getRealName).innerJoin(AuthorDO.class, AuthorDO::getId, UserDO::getAuthorId)
                .eq(AuthorDO::getIsEnable, true).like(AuthorDO::getRealName, authorPojo.getRealName())
                .eq(UserDO::getTypeId, userPojo.getTypeId()).eq(UserDO::getEmail, userPojo.getEmail());
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

    public IPage<UserDO> getUserList(Integer pageNum, UserDO userDO, Integer pageSize) {
        Page<UserDO> page = new Page<>(pageNum, pageSize);
        var wrapper = userDO == null ? null : new MPJLambdaWrapper<>(userDO);
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
    public Long createUser(String email, String password) throws NoSuchAlgorithmException {
        return createUser(email, password, (short) 0);
    }

    @Transactional
    public Long createUser(String email, String password, Short typeId) throws NoSuchAlgorithmException {
        UserDO userDO = new UserDO();
        userDO.setEmail(email);
        userDO.setPassword(md5Encrypt(password));
        userDO.setTypeId(typeId);
        int result = userMapper.createUser(userDO);
        System.out.println("User created with ID: " + userDO.getId() + " and result: " + result);
        // userMapper.selectCount()
        return userDO.getId();
    }

    @Transactional
    public Long createUser(String email, String password, Short typeId, String walletAddress)
            throws NoSuchAlgorithmException {
        UserDO userDO = new UserDO();
        userDO.setId(IdWorker.getId());
        userDO.setEmail(email);
        userDO.setPassword(md5Encrypt(password));
        userDO.setTypeId(typeId);
        userDO.setWalletAddress(walletAddress);
        int result = userMapper.createUserWithWallet(userDO);
        System.out.println("User created with ID: " + userDO.getId() + " and result: " + result);
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

    public boolean deleteUser(Long id) {
        return userMapper.deleteById(id) == 1;
    }

    public boolean deleteUsers(Long[] ids) {
        boolean result = true;
        userMapper.deleteById(ids);
        return result;
    }

    public UserDO validateUser(String email, String password, Short typeId) throws NoSuchAlgorithmException {
        // 根据邮箱和类型ID查询用户
        UserDO user = userMapper.findByEmailAndTypeId(email, typeId);
        if (user == null) {
            return null;
        }
        String encryptedPassword = md5Encrypt(password);
        if (!user.getPassword().equals(encryptedPassword)) {
            return null;
        }

        return user;
    }

    public UserDO findOrCreateUserByWallet(String walletAddress) {
        UserDO user = userMapper.findByWalletAddress(walletAddress);
        if (user != null) {
            return user;
        }
        user = new UserDO();
        // user.setEmail(walletAddress + "@wallet.local"); // generate visual email address
        // user.setPassword("");
        user.setTypeId((short) 1);
        Web3UserDO web3UserDO = Web3UserDO.builder().id(IdWorker.getId()).userId(user.getId()).walletAddress(walletAddress).build();
        // web3UserMapper.insert(web3UserDO);

        try {
            Long userId = createUser(user.getEmail(), user.getPassword(), user.getTypeId(), walletAddress);
            user.setId(userId);
            return user;
        } catch (NoSuchAlgorithmException e) {
            return null;
        }
    }
}
