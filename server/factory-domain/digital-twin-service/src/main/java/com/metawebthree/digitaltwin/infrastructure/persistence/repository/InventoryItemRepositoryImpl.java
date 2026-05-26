package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.digitaltwin.domain.entity.InventoryItem;
import com.metawebthree.digitaltwin.domain.entity.InventoryItem.ItemStatus;
import com.metawebthree.digitaltwin.domain.repository.InventoryItemRepository;
import com.metawebthree.digitaltwin.infrastructure.persistence.converter.InventoryItemConverter;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.InventoryItemDO;
import com.metawebthree.digitaltwin.infrastructure.persistence.mapper.InventoryItemMapper;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class InventoryItemRepositoryImpl implements InventoryItemRepository {

    private final InventoryItemMapper inventoryItemMapper;
    private final InventoryItemConverter inventoryItemConverter;

    public InventoryItemRepositoryImpl(InventoryItemMapper inventoryItemMapper,
                                        InventoryItemConverter inventoryItemConverter) {
        this.inventoryItemMapper = inventoryItemMapper;
        this.inventoryItemConverter = inventoryItemConverter;
    }

    @Override
    public Optional<InventoryItem> findById(Long id) {
        InventoryItemDO itemDO = inventoryItemMapper.selectById(id);
        return Optional.ofNullable(inventoryItemConverter.toEntity(itemDO));
    }

    @Override
    public Optional<InventoryItem> findByItemCode(String itemCode) {
        LambdaQueryWrapper<InventoryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryItemDO::getItemCode, itemCode);
        InventoryItemDO itemDO = inventoryItemMapper.selectOne(wrapper);
        return Optional.ofNullable(inventoryItemConverter.toEntity(itemDO));
    }

    @Override
    public Optional<InventoryItem> findBySku(String sku) {
        LambdaQueryWrapper<InventoryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryItemDO::getSku, sku);
        InventoryItemDO itemDO = inventoryItemMapper.selectOne(wrapper);
        return Optional.ofNullable(inventoryItemConverter.toEntity(itemDO));
    }

    @Override
    public List<InventoryItem> findAll() {
        return inventoryItemMapper.selectList(null).stream()
                .map(inventoryItemConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryItem> findByShelfCode(String shelfCode) {
        LambdaQueryWrapper<InventoryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryItemDO::getShelfCode, shelfCode);
        return inventoryItemMapper.selectList(wrapper).stream()
                .map(inventoryItemConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryItem> findByStatus(ItemStatus status) {
        LambdaQueryWrapper<InventoryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryItemDO::getStatus, status.name());
        return inventoryItemMapper.selectList(wrapper).stream()
                .map(inventoryItemConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryItem> findByCategory(String category) {
        LambdaQueryWrapper<InventoryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryItemDO::getCategory, category);
        return inventoryItemMapper.selectList(wrapper).stream()
                .map(inventoryItemConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryItem> findLowStockItems() {
        LambdaQueryWrapper<InventoryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.isNotNull(InventoryItemDO::getMinQuantity)
               .apply("quantity <= min_quantity");
        return inventoryItemMapper.selectList(wrapper).stream()
                .map(inventoryItemConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryItem> findExpiringSoonItems(int daysThreshold) {
        LocalDate thresholdDate = LocalDate.now().plusDays(daysThreshold);
        LambdaQueryWrapper<InventoryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.isNotNull(InventoryItemDO::getExpiryDate)
               .le(InventoryItemDO::getExpiryDate, thresholdDate)
               .gt(InventoryItemDO::getQuantity, BigDecimal.ZERO);
        return inventoryItemMapper.selectList(wrapper).stream()
                .map(inventoryItemConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public void insert(InventoryItem inventoryItem) {
        InventoryItemDO itemDO = inventoryItemConverter.toDO(inventoryItem);
        inventoryItemMapper.insert(itemDO);
        inventoryItem.setId(itemDO.getId());
    }

    @Override
    public void update(InventoryItem inventoryItem) {
        InventoryItemDO itemDO = inventoryItemConverter.toDO(inventoryItem);
        inventoryItemMapper.updateById(itemDO);
    }

    @Override
    public void delete(InventoryItem inventoryItem) {
        if (inventoryItem.getId() != null) {
            inventoryItemMapper.deleteById(inventoryItem.getId());
        }
    }

    @Override
    public boolean existsByItemCode(String itemCode) {
        LambdaQueryWrapper<InventoryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryItemDO::getItemCode, itemCode);
        return inventoryItemMapper.selectCount(wrapper) > 0;
    }

    @Override
    public boolean existsBySku(String sku) {
        LambdaQueryWrapper<InventoryItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryItemDO::getSku, sku);
        return inventoryItemMapper.selectCount(wrapper) > 0;
    }
}