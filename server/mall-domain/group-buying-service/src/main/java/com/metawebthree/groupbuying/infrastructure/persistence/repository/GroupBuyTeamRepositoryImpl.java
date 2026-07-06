package com.metawebthree.groupbuying.infrastructure.persistence.repository;

import com.metawebthree.groupbuying.domain.model.GroupBuyTeamDO;
import com.metawebthree.groupbuying.domain.repository.GroupBuyTeamRepository;
import com.metawebthree.groupbuying.infrastructure.persistence.mapper.GroupBuyTeamMapper;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class GroupBuyTeamRepositoryImpl implements GroupBuyTeamRepository {

    private final GroupBuyTeamMapper teamMapper;

    public GroupBuyTeamRepositoryImpl(GroupBuyTeamMapper teamMapper) {
        this.teamMapper = teamMapper;
    }

    @Override
    public void save(GroupBuyTeamDO team) {
        teamMapper.insert(team);
    }

    @Override
    public void update(GroupBuyTeamDO team) {
        teamMapper.updateById(team);
    }

    @Override
    public GroupBuyTeamDO findById(Long id) {
        return teamMapper.selectById(id);
    }

    @Override
    public List<GroupBuyTeamDO> findByActivityId(Long activityId) {
        return teamMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyTeamDO>()
                .eq(GroupBuyTeamDO::getActivityId, activityId)
        );
    }

    @Override
    public GroupBuyTeamDO findByTeamNo(String teamNo) {
        return teamMapper.selectOne(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyTeamDO>()
                .eq(GroupBuyTeamDO::getTeamNo, teamNo)
        );
    }

    @Override
    public List<GroupBuyTeamDO> findByLeaderId(Long leaderId) {
        return teamMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyTeamDO>()
                .eq(GroupBuyTeamDO::getLeaderId, leaderId)
        );
    }

    @Override
    public List<GroupBuyTeamDO> findByStatus(String status) {
        return teamMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyTeamDO>()
                .eq(GroupBuyTeamDO::getStatus, status)
        );
    }

    @Override
    public void delete(Long id) {
        teamMapper.deleteById(id);
    }
}