package com.metawebthree.user;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
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
//        userMapper.selectCount()
        return userDO.getId();
    }

    @Transactional
    public Long createUser(String email, String password, Short typeId, String walletAddress) throws NoSuchAlgorithmException {
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

    public boolean deleteUser(Integer id) {
        return userMapper.deleteUser(id);
    }

    public boolean deleteUsers(Integer[] ids) {
        boolean result = true;
        userMapper.deleteUsers(ids);
        return result;
    }

    /**
     * 验证用户登录凭据
     * @param email 用户邮箱
     * @param password 用户密码
     * @param typeId 用户类型ID
     * @return 验证成功返回用户对象，失败返回null
     */
    public UserDO validateUser(String email, String password, Short typeId) throws NoSuchAlgorithmException {
        // 根据邮箱和类型ID查询用户
        UserDO user = userMapper.findByEmailAndTypeId(email, typeId);
        if (user == null) {
            return null;
        }
        
        // 验证密码
        String encryptedPassword = md5Encrypt(password);
        if (!user.getPassword().equals(encryptedPassword)) {
            return null;
        }
        
        return user;
    }

    /**
     * 根据钱包地址查找或创建用户
     * @param walletAddress 钱包地址
     * @return 用户对象
     */
    public UserDO findOrCreateUserByWallet(String walletAddress) {
        // 先尝试查找现有用户
        UserDO user = userMapper.findByWalletAddress(walletAddress);
        if (user != null) {
            return user;
        }
        
        // 如果用户不存在，创建新用户
        user = new UserDO();
        user.setEmail(walletAddress + "@wallet.local"); // 使用钱包地址作为邮箱前缀
        user.setPassword(""); // 钱包用户不需要密码
        user.setTypeId((short) 1); // 设置钱包用户类型
        user.setWalletAddress(walletAddress);
        
        try {
            int userId = createUser(user.getEmail(), user.getPassword(), user.getTypeId(), walletAddress);
            user.setId(userId);
            return user;
        } catch (NoSuchAlgorithmException e) {
            return null;
        }
    }
}
