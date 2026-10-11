package com.metawebthree.recommendation.infrastructure.persistence.repository;

import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.infrastructure.persistence.entity.UserBehaviorDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.UserBehaviorMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.stereotype.Repository;

@Repository
public class UserBehaviorRepositoryImpl implements UserBehaviorRepository {

    private final UserBehaviorMapper userBehaviorMapper;

    public UserBehaviorRepositoryImpl(UserBehaviorMapper userBehaviorMapper) {
        this.userBehaviorMapper = userBehaviorMapper;
    }

    @Override
    public UserBehavior save(UserBehavior userBehavior) {
        UserBehaviorDO userBehaviorDO = toDO(userBehavior);
        if (userBehavior.getId() == null) {
            userBehaviorMapper.insert(userBehaviorDO);
            userBehavior.setId(userBehaviorDO.getId());
        } else {
            userBehaviorMapper.updateById(userBehaviorDO);
        }
        return userBehavior;
    }

    @Override
    public List<UserBehavior> findByUserIdOrderByTimestampDesc(Long userId) {
        LambdaQueryWrapper<UserBehaviorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(UserBehaviorDO::getUserId, userId)
            .orderByDesc(UserBehaviorDO::getTimestamp);
        return userBehaviorMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<UserBehavior> findByProductIdOrderByTimestampDesc(Long productId) {
        LambdaQueryWrapper<UserBehaviorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(UserBehaviorDO::getProductId, productId)
            .orderByDesc(UserBehaviorDO::getTimestamp);
        return userBehaviorMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<UserBehavior> findByUserIdAndBehaviorTypeOrderByTimestampDesc(Long userId, UserBehavior.BehaviorType behaviorType) {
        LambdaQueryWrapper<UserBehaviorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(UserBehaviorDO::getUserId, userId)
            .eq(UserBehaviorDO::getBehaviorType, behaviorType.name())
            .orderByDesc(UserBehaviorDO::getTimestamp);
        return userBehaviorMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<UserBehavior> findRecentByUserId(Long userId, LocalDateTime since) {
        LambdaQueryWrapper<UserBehaviorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(UserBehaviorDO::getUserId, userId)
            .ge(UserBehaviorDO::getTimestamp, since)
            .orderByDesc(UserBehaviorDO::getTimestamp);
        return userBehaviorMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<Long> findUserIdsByProductId(Long productId) {
        LambdaQueryWrapper<UserBehaviorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(UserBehaviorDO::getProductId, productId);
        return userBehaviorMapper.selectList(wrapper).stream()
            .map(UserBehaviorDO::getUserId)
            .distinct()
            .collect(Collectors.toList());
    }

    @Override
    public Long countBehavior(Long userId, Long productId, UserBehavior.BehaviorType behaviorType) {
        LambdaQueryWrapper<UserBehaviorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(UserBehaviorDO::getUserId, userId)
            .eq(UserBehaviorDO::getProductId, productId)
            .eq(UserBehaviorDO::getBehaviorType, behaviorType.name());
        return userBehaviorMapper.selectCount(wrapper);
    }

    @Override
    public void deleteByTimestampBefore(LocalDateTime timestamp) {
        LambdaQueryWrapper<UserBehaviorDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.lt(UserBehaviorDO::getTimestamp, timestamp);
        userBehaviorMapper.delete(wrapper);
    }

    @Override
    public List<UserBehavior> findAll() {
        return userBehaviorMapper.selectList(new LambdaQueryWrapper<>()).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public void deleteById(Long id) {
        userBehaviorMapper.deleteById(id);
    }

    private UserBehavior toDomain(UserBehaviorDO userBehaviorDO) {
        UserBehavior userBehavior = new UserBehavior();
        userBehavior.setId(userBehaviorDO.getId());
        userBehavior.setUserId(userBehaviorDO.getUserId());
        userBehavior.setProductId(userBehaviorDO.getProductId());
        userBehavior.setBehaviorType(UserBehavior.BehaviorType.valueOf(userBehaviorDO.getBehaviorType()));
        userBehavior.setBehaviorValue(userBehaviorDO.getBehaviorValue());
        userBehavior.setTimestamp(userBehaviorDO.getTimestamp());
        userBehavior.setSessionId(userBehaviorDO.getSessionId());
        userBehavior.setSource(userBehaviorDO.getSource());
        return userBehavior;
    }

    private UserBehaviorDO toDO(UserBehavior userBehavior) {
        UserBehaviorDO userBehaviorDO = new UserBehaviorDO();
        userBehaviorDO.setId(userBehavior.getId());
        userBehaviorDO.setUserId(userBehavior.getUserId());
        userBehaviorDO.setProductId(userBehavior.getProductId());
        userBehaviorDO.setBehaviorType(userBehavior.getBehaviorType().name());
        userBehaviorDO.setBehaviorValue(userBehavior.getBehaviorValue());
        userBehaviorDO.setTimestamp(userBehavior.getTimestamp());
        userBehaviorDO.setSessionId(userBehavior.getSessionId());
        userBehaviorDO.setSource(userBehavior.getSource());
        return userBehaviorDO;
    }
}
