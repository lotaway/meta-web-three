package com.metawebthree.logistics.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.logistics.domain.entity.LogisticsOrder;
import com.metawebthree.logistics.infrastructure.persistence.converter.LogisticsOrderConverter;
import com.metawebthree.logistics.infrastructure.persistence.dataobject.LogisticsOrderDO;
import com.metawebthree.logistics.infrastructure.persistence.mapper.LogisticsOrderMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class LogisticsOrderRepositoryImpl implements LogisticsOrderRepository {

    private final LogisticsOrderMapper logisticsOrderMapper;
    private final LogisticsOrderConverter logisticsOrderConverter;

    public LogisticsOrderRepositoryImpl(LogisticsOrderMapper logisticsOrderMapper, 
                                         LogisticsOrderConverter logisticsOrderConverter) {
        this.logisticsOrderMapper = logisticsOrderMapper;
        this.logisticsOrderConverter = logisticsOrderConverter;
    }

    @Override
    public Optional<LogisticsOrder> findById(Long id) {
        LogisticsOrderDO orderDO = logisticsOrderMapper.selectById(id);
        return Optional.ofNullable(logisticsOrderConverter.toEntity(orderDO));
    }

    @Override
    public Optional<LogisticsOrder> findByTrackingNo(String trackingNo) {
        LambdaQueryWrapper<LogisticsOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(LogisticsOrderDO::getTrackingNo, trackingNo);
        LogisticsOrderDO orderDO = logisticsOrderMapper.selectOne(wrapper);
        return Optional.ofNullable(logisticsOrderConverter.toEntity(orderDO));
    }

    @Override
    public Optional<LogisticsOrder> findByOrderNo(String orderNo) {
        LambdaQueryWrapper<LogisticsOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(LogisticsOrderDO::getOrderNo, orderNo);
        LogisticsOrderDO orderDO = logisticsOrderMapper.selectOne(wrapper);
        return Optional.ofNullable(logisticsOrderConverter.toEntity(orderDO));
    }

    @Override
    public List<LogisticsOrder> findAll() {
        List<LogisticsOrderDO> orderDOs = logisticsOrderMapper.selectList(null);
        return orderDOs.stream()
            .map(logisticsOrderConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<LogisticsOrder> findByCarrierId(Long carrierId) {
        LambdaQueryWrapper<LogisticsOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(LogisticsOrderDO::getCarrierId, carrierId);
        List<LogisticsOrderDO> orderDOs = logisticsOrderMapper.selectList(wrapper);
        return orderDOs.stream()
            .map(logisticsOrderConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<LogisticsOrder> findByStatus(String status) {
        LambdaQueryWrapper<LogisticsOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(LogisticsOrderDO::getStatus, status);
        List<LogisticsOrderDO> orderDOs = logisticsOrderMapper.selectList(wrapper);
        return orderDOs.stream()
            .map(logisticsOrderConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<LogisticsOrder> findByCarrierIdAndStatus(Long carrierId, String status) {
        LambdaQueryWrapper<LogisticsOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(LogisticsOrderDO::getCarrierId, carrierId)
               .eq(LogisticsOrderDO::getStatus, status);
        List<LogisticsOrderDO> orderDOs = logisticsOrderMapper.selectList(wrapper);
        return orderDOs.stream()
            .map(logisticsOrderConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public LogisticsOrder save(LogisticsOrder order) {
        LogisticsOrderDO orderDO = logisticsOrderConverter.toDO(order);
        if (order.getId() == null) {
            logisticsOrderMapper.insert(orderDO);
            order.setId(orderDO.getId());
        } else {
            logisticsOrderMapper.updateById(orderDO);
        }
        return order;
    }

    @Override
    public void delete(LogisticsOrder order) {
        if (order.getId() != null) {
            logisticsOrderMapper.deleteById(order.getId());
        }
    }
}