package com.metawebthree.warehouse.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.warehouse.domain.entity.InboundOrder;
import com.metawebthree.warehouse.domain.entity.InboundOrderItem;
import com.metawebthree.warehouse.infrastructure.persistence.converter.InboundOrderConverter;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.InboundOrderDO;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.InboundOrderItemDO;
import com.metawebthree.warehouse.infrastructure.persistence.mapper.InboundOrderItemMapper;
import com.metawebthree.warehouse.infrastructure.persistence.mapper.InboundOrderMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class InboundOrderRepositoryImpl implements InboundOrderRepository {

    private final InboundOrderMapper inboundOrderMapper;
    private final InboundOrderItemMapper inboundOrderItemMapper;
    private final InboundOrderConverter inboundOrderConverter;

    public InboundOrderRepositoryImpl(InboundOrderMapper inboundOrderMapper,
                                      InboundOrderItemMapper inboundOrderItemMapper,
                                      InboundOrderConverter inboundOrderConverter) {
        this.inboundOrderMapper = inboundOrderMapper;
        this.inboundOrderItemMapper = inboundOrderItemMapper;
        this.inboundOrderConverter = inboundOrderConverter;
    }

    @Override
    public Optional<InboundOrder> findById(Long id) {
        InboundOrderDO orderDO = inboundOrderMapper.selectById(id);
        if (orderDO == null) {
            return Optional.empty();
        }
        InboundOrder order = inboundOrderConverter.toEntity(orderDO);
        loadItems(order);
        return Optional.of(order);
    }

    @Override
    public Optional<InboundOrder> findByOrderNo(String orderNo) {
        LambdaQueryWrapper<InboundOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InboundOrderDO::getOrderNo, orderNo);
        InboundOrderDO orderDO = inboundOrderMapper.selectOne(wrapper);
        if (orderDO == null) {
            return Optional.empty();
        }
        InboundOrder order = inboundOrderConverter.toEntity(orderDO);
        loadItems(order);
        return Optional.of(order);
    }

    @Override
    public void insert(InboundOrder order) {
        InboundOrderDO orderDO = inboundOrderConverter.toDO(order);
        inboundOrderMapper.insert(orderDO);
        order.setId(orderDO.getId());
        
        if (order.getItems() != null && !order.getItems().isEmpty()) {
            for (InboundOrderItem item : order.getItems()) {
                item.setOrderId(orderDO.getId());
                InboundOrderItemDO itemDO = inboundOrderConverter.toDOItem(item);
                inboundOrderItemMapper.insert(itemDO);
                item.setId(itemDO.getId());
            }
        }
    }

    @Override
    public void update(InboundOrder order) {
        InboundOrderDO orderDO = inboundOrderConverter.toDO(order);
        inboundOrderMapper.updateById(orderDO);
    }

    @Override
    public List<InboundOrder> findByWarehouseId(Long warehouseId) {
        LambdaQueryWrapper<InboundOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InboundOrderDO::getWarehouseId, warehouseId);
        return inboundOrderMapper.selectList(wrapper).stream()
                .map(doObj -> {
                    InboundOrder order = inboundOrderConverter.toEntity(doObj);
                    loadItems(order);
                    return order;
                })
                .collect(Collectors.toList());
    }

    @Override
    public List<InboundOrder> findByStatus(String status) {
        LambdaQueryWrapper<InboundOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InboundOrderDO::getStatus, status);
        return inboundOrderMapper.selectList(wrapper).stream()
                .map(doObj -> {
                    InboundOrder order = inboundOrderConverter.toEntity(doObj);
                    loadItems(order);
                    return order;
                })
                .collect(Collectors.toList());
    }

    private void loadItems(InboundOrder order) {
        if (order.getId() == null) {
            return;
        }
        LambdaQueryWrapper<InboundOrderItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InboundOrderItemDO::getOrderId, order.getId());
        List<InboundOrderItem> items = inboundOrderItemMapper.selectList(wrapper).stream()
                .map(inboundOrderConverter::toEntityItem)
                .collect(Collectors.toList());
        order.setItems(items);
    }
}