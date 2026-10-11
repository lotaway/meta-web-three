package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.digitaltwin.domain.entity.Shelf;
import com.metawebthree.digitaltwin.domain.entity.Shelf.ShelfStatus;
import com.metawebthree.digitaltwin.domain.repository.ShelfRepository;
import com.metawebthree.digitaltwin.infrastructure.persistence.converter.ShelfConverter;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.ShelfDO;
import com.metawebthree.digitaltwin.infrastructure.persistence.mapper.ShelfMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class ShelfRepositoryImpl implements ShelfRepository {

    private final ShelfMapper shelfMapper;
    private final ShelfConverter shelfConverter;

    public ShelfRepositoryImpl(ShelfMapper shelfMapper, ShelfConverter shelfConverter) {
        this.shelfMapper = shelfMapper;
        this.shelfConverter = shelfConverter;
    }

    @Override
    public Optional<Shelf> findById(Long id) {
        ShelfDO shelfDO = shelfMapper.selectById(id);
        return Optional.ofNullable(shelfConverter.toEntity(shelfDO));
    }

    @Override
    public Optional<Shelf> findByShelfCode(String shelfCode) {
        LambdaQueryWrapper<ShelfDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ShelfDO::getShelfCode, shelfCode);
        ShelfDO shelfDO = shelfMapper.selectOne(wrapper);
        return Optional.ofNullable(shelfConverter.toEntity(shelfDO));
    }

    @Override
    public List<Shelf> findAll() {
        return shelfMapper.selectList(null).stream()
                .map(shelfConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Shelf> findByWarehouseCode(String warehouseCode) {
        LambdaQueryWrapper<ShelfDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ShelfDO::getWarehouseCode, warehouseCode);
        return shelfMapper.selectList(wrapper).stream()
                .map(shelfConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Shelf> findByWarehouseCodeAndStatus(String warehouseCode, ShelfStatus status) {
        LambdaQueryWrapper<ShelfDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ShelfDO::getWarehouseCode, warehouseCode)
               .eq(ShelfDO::getStatus, status.name());
        return shelfMapper.selectList(wrapper).stream()
                .map(shelfConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Shelf> findByWarehouseCodeAndZone(String warehouseCode, String zone) {
        LambdaQueryWrapper<ShelfDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ShelfDO::getWarehouseCode, warehouseCode)
               .eq(ShelfDO::getZone, zone);
        return shelfMapper.selectList(wrapper).stream()
                .map(shelfConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Shelf> findEmptyShelves(String warehouseCode) {
        return findByWarehouseCodeAndStatus(warehouseCode, ShelfStatus.EMPTY);
    }

    @Override
    public void insert(Shelf shelf) {
        ShelfDO shelfDO = shelfConverter.toDO(shelf);
        shelfMapper.insert(shelfDO);
        shelf.setId(shelfDO.getId());
    }

    @Override
    public void update(Shelf shelf) {
        ShelfDO shelfDO = shelfConverter.toDO(shelf);
        shelfMapper.updateById(shelfDO);
    }

    @Override
    public void delete(Shelf shelf) {
        if (shelf.getId() != null) {
            shelfMapper.deleteById(shelf.getId());
        }
    }

    @Override
    public boolean existsByShelfCode(String shelfCode) {
        LambdaQueryWrapper<ShelfDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ShelfDO::getShelfCode, shelfCode);
        return shelfMapper.selectCount(wrapper) > 0;
    }
}