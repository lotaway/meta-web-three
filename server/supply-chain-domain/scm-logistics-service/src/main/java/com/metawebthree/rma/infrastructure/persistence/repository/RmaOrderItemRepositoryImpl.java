package com.metawebthree.rma.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.rma.domain.entity.RmaOrderItem;
import com.metawebthree.rma.domain.repository.RmaOrderItemRepository;
import com.metawebthree.rma.infrastructure.persistence.converter.RmaOrderItemConverter;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaOrderItemDO;
import com.metawebthree.rma.infrastructure.persistence.mapper.RmaOrderItemMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.stream.Collectors;

@Repository
public class RmaOrderItemRepositoryImpl implements RmaOrderItemRepository {

    private final RmaOrderItemMapper rmaOrderItemMapper;
    private final RmaOrderItemConverter rmaOrderItemConverter;

    public RmaOrderItemRepositoryImpl(RmaOrderItemMapper rmaOrderItemMapper,
                                      RmaOrderItemConverter rmaOrderItemConverter) {
        this.rmaOrderItemMapper = rmaOrderItemMapper;
        this.rmaOrderItemConverter = rmaOrderItemConverter;
    }

    @Override
    public List<RmaOrderItem> findByRmaId(Long rmaId) {
        LambdaQueryWrapper<RmaOrderItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RmaOrderItemDO::getRmaId, rmaId);
        return rmaOrderItemMapper.selectList(wrapper).stream()
                .map(rmaOrderItemConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public RmaOrderItem save(RmaOrderItem item) {
        RmaOrderItemDO doObj = rmaOrderItemConverter.toDO(item);
        if (item.getId() == null) {
            rmaOrderItemMapper.insert(doObj);
            item.setId(doObj.getId());
        } else {
            rmaOrderItemMapper.updateById(doObj);
        }
        return item;
    }

    @Override
    public List<RmaOrderItem> saveAll(List<RmaOrderItem> items) {
        return items.stream().map(this::save).collect(Collectors.toList());
    }

    @Override
    public void deleteByRmaId(Long rmaId) {
        LambdaQueryWrapper<RmaOrderItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RmaOrderItemDO::getRmaId, rmaId);
        rmaOrderItemMapper.delete(wrapper);
    }
}
