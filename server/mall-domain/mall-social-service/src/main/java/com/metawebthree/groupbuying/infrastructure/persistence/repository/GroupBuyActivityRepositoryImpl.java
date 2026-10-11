package com.metawebthree.groupbuying.infrastructure.persistence.repository;

import com.metawebthree.groupbuying.domain.model.GroupBuyActivityDO;
import com.metawebthree.groupbuying.domain.repository.GroupBuyActivityRepository;
import com.metawebthree.groupbuying.infrastructure.persistence.mapper.GroupBuyActivityMapper;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class GroupBuyActivityRepositoryImpl implements GroupBuyActivityRepository {

    private final GroupBuyActivityMapper activityMapper;

    public GroupBuyActivityRepositoryImpl(GroupBuyActivityMapper activityMapper) {
        this.activityMapper = activityMapper;
    }

    @Override
    public void save(GroupBuyActivityDO activity) {
        activityMapper.insert(activity);
    }

    @Override
    public void update(GroupBuyActivityDO activity) {
        activityMapper.updateById(activity);
    }

    @Override
    public GroupBuyActivityDO findById(Long id) {
        return activityMapper.selectById(id);
    }

    @Override
    public List<GroupBuyActivityDO> findByStatus(Integer status) {
        return activityMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyActivityDO>()
                .eq(GroupBuyActivityDO::getStatus, status)
        );
    }

    @Override
    public List<GroupBuyActivityDO> findActiveActivities() {
        return activityMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyActivityDO>()
                .eq(GroupBuyActivityDO::getStatus, 1)
        );
    }

    @Override
    public void delete(Long id) {
        activityMapper.deleteById(id);
    }
}