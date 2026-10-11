package com.metawebthree.settlement.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.settlement.domain.entity.SettlementOrder;
import com.metawebthree.settlement.domain.repository.SettlementOrderRepository;
import com.metawebthree.settlement.infrastructure.persistence.converter.SettlementOrderConverter;
import com.metawebthree.settlement.infrastructure.persistence.dataobject.SettlementOrderDO;
import com.metawebthree.settlement.infrastructure.persistence.mapper.SettlementOrderMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class SettlementOrderRepositoryImpl implements SettlementOrderRepository {

    private final SettlementOrderMapper mapper;
    private final SettlementOrderConverter converter;

    public SettlementOrderRepositoryImpl(SettlementOrderMapper mapper, SettlementOrderConverter converter) {
        this.mapper = mapper;
        this.converter = converter;
    }

    @Override
    public Optional<SettlementOrder> findById(Long id) {
        return Optional.ofNullable(converter.toEntity(mapper.selectById(id)));
    }

    @Override
    public Optional<SettlementOrder> findBySettlementNo(String settlementNo) {
        LambdaQueryWrapper<SettlementOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SettlementOrderDO::getSettlementNo, settlementNo);
        return Optional.ofNullable(converter.toEntity(mapper.selectOne(wrapper)));
    }

    @Override
    public List<SettlementOrder> findByStatus(SettlementOrder.SettlementStatus status) {
        LambdaQueryWrapper<SettlementOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SettlementOrderDO::getStatus, status.name());
        return mapper.selectList(wrapper).stream().map(converter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<SettlementOrder> findByMerchantId(Long merchantId) {
        LambdaQueryWrapper<SettlementOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SettlementOrderDO::getMerchantId, merchantId);
        return mapper.selectList(wrapper).stream().map(converter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<SettlementOrder> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end) {
        LambdaQueryWrapper<SettlementOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(SettlementOrderDO::getSettlementDate, start, end);
        return mapper.selectList(wrapper).stream().map(converter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<SettlementOrder> findAll() {
        return mapper.selectList(null).stream().map(converter::toEntity).collect(Collectors.toList());
    }

    @Override
    public void save(SettlementOrder order) {
        SettlementOrderDO doObj = converter.toDO(order);
        if (order.getId() == null) {
            mapper.insert(doObj);
            order.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
    }

    @Override
    public void update(SettlementOrder order) {
        mapper.updateById(converter.toDO(order));
    }

    @Override
    public void delete(Long id) {
        mapper.deleteById(id);
    }
}