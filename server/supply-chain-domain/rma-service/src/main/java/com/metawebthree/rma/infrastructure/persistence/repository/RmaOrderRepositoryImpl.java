package com.metawebthree.rma.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.rma.domain.entity.RmaOrder;
import com.metawebthree.rma.domain.repository.RmaOrderRepository;
import com.metawebthree.rma.infrastructure.persistence.converter.RmaOrderConverter;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaOrderDO;
import com.metawebthree.rma.infrastructure.persistence.mapper.RmaOrderMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class RmaOrderRepositoryImpl implements RmaOrderRepository {

    private final RmaOrderMapper rmaOrderMapper;
    private final RmaOrderConverter rmaOrderConverter;

    public RmaOrderRepositoryImpl(RmaOrderMapper rmaOrderMapper, RmaOrderConverter rmaOrderConverter) {
        this.rmaOrderMapper = rmaOrderMapper;
        this.rmaOrderConverter = rmaOrderConverter;
    }

    @Override
    public Optional<RmaOrder> findById(Long id) {
        RmaOrderDO doObj = rmaOrderMapper.selectById(id);
        return Optional.ofNullable(rmaOrderConverter.toEntity(doObj));
    }

    @Override
    public Optional<RmaOrder> findByRmaNo(String rmaNo) {
        LambdaQueryWrapper<RmaOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RmaOrderDO::getRmaNo, rmaNo);
        RmaOrderDO doObj = rmaOrderMapper.selectOne(wrapper);
        return Optional.ofNullable(rmaOrderConverter.toEntity(doObj));
    }

    @Override
    public List<RmaOrder> findByOrderNo(String orderNo) {
        LambdaQueryWrapper<RmaOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RmaOrderDO::getOrderNo, orderNo);
        return rmaOrderMapper.selectList(wrapper).stream()
                .map(rmaOrderConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<RmaOrder> findByStatus(String status) {
        LambdaQueryWrapper<RmaOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RmaOrderDO::getStatus, status);
        return rmaOrderMapper.selectList(wrapper).stream()
                .map(rmaOrderConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<RmaOrder> findAll() {
        return rmaOrderMapper.selectList(null).stream()
                .map(rmaOrderConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public IPage<RmaOrder> findPage(Page<RmaOrder> page, String status) {
        Page<RmaOrderDO> doPage = new Page<>(page.getCurrent(), page.getSize());
        LambdaQueryWrapper<RmaOrderDO> wrapper = new LambdaQueryWrapper<>();
        if (status != null && !status.isEmpty()) {
            wrapper.eq(RmaOrderDO::getStatus, status);
        }
        wrapper.orderByDesc(RmaOrderDO::getCreatedAt);
        IPage<RmaOrderDO> doResult = rmaOrderMapper.selectPage(doPage, wrapper);
        IPage<RmaOrder> result = new Page<>(doResult.getCurrent(), doResult.getSize(), doResult.getTotal());
        result.setRecords(doResult.getRecords().stream()
                .map(rmaOrderConverter::toEntity)
                .collect(Collectors.toList()));
        return result;
    }

    @Override
    public RmaOrder save(RmaOrder order) {
        RmaOrderDO doObj = rmaOrderConverter.toDO(order);
        if (order.getId() == null) {
            rmaOrderMapper.insert(doObj);
            order.setId(doObj.getId());
        } else {
            rmaOrderMapper.updateById(doObj);
        }
        return order;
    }
}
