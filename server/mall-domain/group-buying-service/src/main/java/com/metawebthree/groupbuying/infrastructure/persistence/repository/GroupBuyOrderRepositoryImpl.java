package com.metawebthree.groupbuying.infrastructure.persistence.repository;

import com.metawebthree.groupbuying.domain.model.GroupBuyOrderDO;
import com.metawebthree.groupbuying.domain.repository.GroupBuyOrderRepository;
import com.metawebthree.groupbuying.infrastructure.persistence.mapper.GroupBuyOrderMapper;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class GroupBuyOrderRepositoryImpl implements GroupBuyOrderRepository {

    private final GroupBuyOrderMapper orderMapper;

    public GroupBuyOrderRepositoryImpl(GroupBuyOrderMapper orderMapper) {
        this.orderMapper = orderMapper;
    }

    @Override
    public void save(GroupBuyOrderDO order) {
        orderMapper.insert(order);
    }

    @Override
    public void update(GroupBuyOrderDO order) {
        orderMapper.updateById(order);
    }

    @Override
    public GroupBuyOrderDO findById(Long id) {
        return orderMapper.selectById(id);
    }

    @Override
    public GroupBuyOrderDO findByOrderNo(String orderNo) {
        return orderMapper.selectOne(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyOrderDO>()
                .eq(GroupBuyOrderDO::getOrderNo, orderNo)
        );
    }

    @Override
    public GroupBuyOrderDO findByTeamIdAndUserId(Long teamId, Long userId) {
        return orderMapper.selectOne(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyOrderDO>()
                .eq(GroupBuyOrderDO::getTeamId, teamId)
                .eq(GroupBuyOrderDO::getUserId, userId)
        );
    }

    @Override
    public List<GroupBuyOrderDO> findByTeamId(Long teamId) {
        return orderMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyOrderDO>()
                .eq(GroupBuyOrderDO::getTeamId, teamId)
        );
    }

    @Override
    public List<GroupBuyOrderDO> findByUserId(Long userId) {
        return orderMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyOrderDO>()
                .eq(GroupBuyOrderDO::getUserId, userId)
        );
    }

    @Override
    public List<GroupBuyOrderDO> findByStatus(String status) {
        return orderMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<GroupBuyOrderDO>()
                .eq(GroupBuyOrderDO::getStatus, status)
        );
    }

    @Override
    public void delete(Long id) {
        orderMapper.deleteById(id);
    }
}