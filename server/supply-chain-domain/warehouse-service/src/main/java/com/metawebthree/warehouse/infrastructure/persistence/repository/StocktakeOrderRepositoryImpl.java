package com.metawebthree.warehouse.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.warehouse.domain.entity.StocktakeOrder;
import com.metawebthree.warehouse.domain.entity.StocktakeOrderItem;
import com.metawebthree.warehouse.domain.repository.StocktakeOrderRepository;
import com.metawebthree.warehouse.infrastructure.persistence.converter.StocktakeOrderConverter;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.StocktakeOrderDO;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.StocktakeOrderItemDO;
import com.metawebthree.warehouse.infrastructure.persistence.mapper.StocktakeOrderItemMapper;
import com.metawebthree.warehouse.infrastructure.persistence.mapper.StocktakeOrderMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class StocktakeOrderRepositoryImpl implements StocktakeOrderRepository {

    private final StocktakeOrderMapper stocktakeOrderMapper;
    private final StocktakeOrderItemMapper stocktakeOrderItemMapper;
    private final StocktakeOrderConverter stocktakeOrderConverter;

    public StocktakeOrderRepositoryImpl(StocktakeOrderMapper stocktakeOrderMapper,
                                         StocktakeOrderItemMapper stocktakeOrderItemMapper,
                                         StocktakeOrderConverter stocktakeOrderConverter) {
        this.stocktakeOrderMapper = stocktakeOrderMapper;
        this.stocktakeOrderItemMapper = stocktakeOrderItemMapper;
        this.stocktakeOrderConverter = stocktakeOrderConverter;
    }

    @Override
    public Optional<StocktakeOrder> findById(Long id) {
        StocktakeOrderDO orderDO = stocktakeOrderMapper.selectById(id);
        if (orderDO == null) {
            return Optional.empty();
        }
        StocktakeOrder order = stocktakeOrderConverter.toEntity(orderDO);
        loadItems(order);
        return Optional.of(order);
    }

    @Override
    public Optional<StocktakeOrder> findByOrderNo(String orderNo) {
        LambdaQueryWrapper<StocktakeOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StocktakeOrderDO::getOrderNo, orderNo);
        StocktakeOrderDO orderDO = stocktakeOrderMapper.selectOne(wrapper);
        if (orderDO == null) {
            return Optional.empty();
        }
        StocktakeOrder order = stocktakeOrderConverter.toEntity(orderDO);
        loadItems(order);
        return Optional.of(order);
    }

    @Override
    public List<StocktakeOrder> findByWarehouseId(Long warehouseId) {
        LambdaQueryWrapper<StocktakeOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StocktakeOrderDO::getWarehouseId, warehouseId);
        return stocktakeOrderMapper.selectList(wrapper).stream()
                .map(doObj -> {
                    StocktakeOrder order = stocktakeOrderConverter.toEntity(doObj);
                    loadItems(order);
                    return order;
                })
                .collect(Collectors.toList());
    }

    @Override
    public List<StocktakeOrder> findByStatus(String status) {
        LambdaQueryWrapper<StocktakeOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StocktakeOrderDO::getStatus, status);
        return stocktakeOrderMapper.selectList(wrapper).stream()
                .map(doObj -> {
                    StocktakeOrder order = stocktakeOrderConverter.toEntity(doObj);
                    loadItems(order);
                    return order;
                })
                .collect(Collectors.toList());
    }

    @Override
    public List<StocktakeOrder> findByWarehouseIdAndStatus(Long warehouseId, String status) {
        LambdaQueryWrapper<StocktakeOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StocktakeOrderDO::getWarehouseId, warehouseId)
               .eq(StocktakeOrderDO::getStatus, status);
        return stocktakeOrderMapper.selectList(wrapper).stream()
                .map(doObj -> {
                    StocktakeOrder order = stocktakeOrderConverter.toEntity(doObj);
                    loadItems(order);
                    return order;
                })
                .collect(Collectors.toList());
    }

    @Override
    public void insert(StocktakeOrder order) {
        StocktakeOrderDO orderDO = stocktakeOrderConverter.toDO(order);
        stocktakeOrderMapper.insert(orderDO);
        order.setId(orderDO.getId());
        
        if (order.getItems() != null && !order.getItems().isEmpty()) {
            for (StocktakeOrderItem item : order.getItems()) {
                item.setStocktakeOrderId(orderDO.getId());
                StocktakeOrderItemDO itemDO = stocktakeOrderConverter.toDOItem(item);
                stocktakeOrderItemMapper.insert(itemDO);
                item.setId(itemDO.getId());
            }
        }
    }

    @Override
    public void update(StocktakeOrder order) {
        StocktakeOrderDO orderDO = stocktakeOrderConverter.toDO(order);
        stocktakeOrderMapper.updateById(orderDO);
    }

    @Override
    public void delete(Long id) {
        stocktakeOrderMapper.deleteById(id);
    }

    private void loadItems(StocktakeOrder order) {
        if (order.getId() == null) {
            return;
        }
        LambdaQueryWrapper<StocktakeOrderItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StocktakeOrderItemDO::getStocktakeOrderId, order.getId());
        List<StocktakeOrderItem> items = stocktakeOrderItemMapper.selectList(wrapper).stream()
                .map(stocktakeOrderConverter::toEntityItem)
                .collect(Collectors.toList());
        order.setItems(items);
    }
}
