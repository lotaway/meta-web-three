package com.metawebthree.user.impl;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.author.AuthorDO;
import com.metawebthree.common.PageConfigVO;
import com.metawebthree.common.utils.UserRole;
import com.metawebthree.user.UserService;
import com.metawebthree.user.Web3UserMapper;
import com.metawebthree.user.UserMapper;
import com.metawebthree.user.UserRoleMappingMapper;
import com.metawebthree.user.DO.UserDO;
import com.metawebthree.user.DO.UserRoleMappingDO;
import com.metawebthree.user.DO.Web3UserDO;
import com.metawebthree.user.DTO.UserDTO;

import lombok.extern.slf4j.Slf4j;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@Service
public class UserServiceImpl extends ServiceImpl<UserMapper, UserDO> implements UserService {

    private final PageConfigVO pageConfigVo;

    private final UserMapper userMapper;
    private final Web3UserMapper web3UserMapper;
    private final UserRoleMappingMapper userRoleMappingMapper;

    public UserServiceImpl(PageConfigVO pageConfigVo, UserMapper userMapper, Web3UserMapper web3UserMapper,
            UserRoleMappingMapper userRoleMappingMapper) {
        this.pageConfigVo = pageConfigVo;
        this.userMapper = userMapper;
        this.web3UserMapper = web3UserMapper;
        this.userRoleMappingMapper = userRoleMappingMapper;
    }

    public IPage<UserDTO> getUserList(int pageNum, UserDTO userDTO, AuthorDO authorDO) {
        // QueryWrapper<UserPojo> userQueryWrapper = new QueryWrapper<>();
        // userQueryWrapper.select("id", "email", "authorId", "typeId").like("email",
        // email).eq("user_type", user_type).eq("user_type", userType).and(wrapper ->
        // wrapper.inSql("User.author_id", "select author_id from Author where real_name
        // =" + realName));
        MPJLambdaWrapper<UserDO> wrapper = new MPJLambdaWrapper<>();
        wrapper.select(UserDO::getId, UserDO::getEmail)
                .select(AuthorDO::getUserId, AuthorDO::getRealName)
                .select(UserRoleMappingDO::getUserRoleId)
                .select(Web3UserDO::getWalletAddress)
                .leftJoin(
                        AuthorDO.class,
                        AuthorDO::getUserId,
                        UserDO::getId)
                .leftJoin(
                        Web3UserDO.class,
                        Web3UserDO::getUserId,
                        UserDO::getId)
                .leftJoin(
                        UserRoleMappingDO.class,
                        UserRoleMappingDO::getUserId,
                        UserDO::getId)
                .eq(
                        AuthorDO::getIsEnable,
                        true)
                .like(
                        AuthorDO::getRealName,
                        authorDO.getRealName())
                .eq(
                        UserDTO::getUserRoleId,
                        userDTO.getUserRoleId())
                .eq(
                        UserDTO::getEmail,
                        userDTO.getEmail());
        return getUserList(pageNum, wrapper);
    }

    public IPage<UserDTO> getUserList() {
        return getUserList(1);
    }

    public IPage<UserDTO> getUserList(int pageNum) {
        return getUserList(pageNum, pageConfigVo.getPageSize());
    }

    public IPage<UserDTO> getUserList(int pageNum, int pageSize) {
        return getUserList(pageNum, (UserDTO) null, pageSize);
    }

    public IPage<UserDTO> getUserList(Integer pageNum, UserDTO userDTO, Integer pageSize) {
        Page<UserDTO> page = new Page<>(pageNum, pageSize);
        MPJLambdaWrapper<UserDO> wrapper = new MPJLambdaWrapper<>();
        wrapper.selectAll(UserDO.class)
                .select(UserRoleMappingDO::getUserRoleId)
                .select(Web3UserDO::getWalletAddress)
                .leftJoin(UserRoleMappingDO.class, UserRoleMappingDO::getUserId, UserDO::getId)
                .leftJoin(Web3UserDO.class, Web3UserDO::getUserId, UserDO::getId);
        if (userDTO != null) {
            if (userDTO.getEmail() != null) {
                wrapper.eq(UserDO::getEmail, userDTO.getEmail());
            }
            if (userDTO.getUserRoleId() != null) {
                wrapper.eq(UserRoleMappingDO::getUserRoleId, userDTO.getUserRoleId());
            }
        }
        return userMapper.selectJoinPage(page, UserDTO.class, wrapper);
    }

    public IPage<UserDTO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper) {
        return getUserList(pageNum, wrapper, pageConfigVo.getPageSize());
    }

    public IPage<UserDTO> getUserList(Integer pageNum, MPJLambdaWrapper<UserDO> wrapper, Integer pageSize) {
        Page<UserDTO> page = new Page<>(pageNum, pageSize);
        return userMapper.selectJoinPage(page, UserDTO.class, wrapper);
    }

    @Transactional
    public Long createUser(String email, String password) throws NoSuchAlgorithmException {
        return createUser(email, password, UserRole.USER);
    }

    @Transactional
    public Long createUser(String email, String password, UserRole userRoleId) throws NoSuchAlgorithmException {
        UserDO userDO = new UserDO();
        userDO.setEmail(email);
        userDO.setPassword(md5Encrypt(password));
        UserRoleMappingDO userRoleMappingDO = UserRoleMappingDO.builder().userRoleId(userRoleId).build();
        int result = userMapper.createUser(userDO);
        int result2 = userRoleMappingMapper.insert(userRoleMappingDO);
        log.info("User created with ID: " + userDO.getId() + " and result: " + result + ", result2: " + result2);
        // userMapper.selectCount()
        return userDO.getId();
    }

    @Transactional
    public Long createUser(String email, String password, UserRole userRoleId, String walletAddress)
            throws NoSuchAlgorithmException {
        UserDO userDO = new UserDO();
        userDO.setId(IdWorker.getId());
        userDO.setEmail(email);
        userDO.setPassword(md5Encrypt(password));
        UserRoleMappingDO userRoleMappingDO = UserRoleMappingDO
                .builder()
                .id(IdWorker.getId())
                .userId(userDO.getId())
                .userRoleId(userRoleId)
                .build();
        Web3UserDO web3UserDO = Web3UserDO
                .builder()
                .id(IdWorker.getId())
                .userId(userDO.getId())
                .walletAddress(walletAddress)
                .build();
        // int result = userMapper.createUserWithWallet(userDO);
        int result = userMapper.insert(userDO);
        int result2 = userRoleMappingMapper.insert(userRoleMappingDO);
        int result3 = web3UserMapper.insert(web3UserDO);
        log.info("User created with ID: " + userDO.getId() + " and result: " + result + ", result2: " + result2
                + ", result3: " + result3);
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

    public UserDTO validateUser(String email, String password, Long userRoleId) throws NoSuchAlgorithmException {
        // 根据邮箱和类型ID查询用户
        UserDTO user = userMapper.findByEmailAndTypeId(email, userRoleId);
        if (user == null) {
            return null;
        }
        String encryptedPassword = md5Encrypt(password);
        if (!user.getPassword().equals(encryptedPassword)) {
            return null;
        }

        return user;
    }

    public UserDTO findOrCreateUserByWallet(String walletAddress) {
        UserDTO user = userMapper.findByWalletAddress(walletAddress);
        if (user != null) {
            return user;
        }
        user = new UserDTO();
        // user.setEmail(walletAddress + "@wallet.local"); // generate visual email
        // address
        // user.setPassword("");
        user.setUserRoleId(UserRole.USER);
        Web3UserDO web3UserDO = Web3UserDO.builder().id(IdWorker.getId()).userId(user.getId())
                .walletAddress(walletAddress).build();
        web3UserMapper.insert(web3UserDO);

        try {
            Long userId = createUser(user.getEmail(), user.getPassword(), user.getUserRoleId(), walletAddress);
            user.setId(userId);
            return user;
        } catch (NoSuchAlgorithmException e) {
            return null;
        }
    }
}
