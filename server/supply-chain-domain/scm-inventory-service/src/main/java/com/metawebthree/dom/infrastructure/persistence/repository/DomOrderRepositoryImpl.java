package com.metawebthree.dom.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.dom.domain.entity.DomOrder;
import com.metawebthree.dom.domain.entity.DomOrderStatus;
import com.metawebthree.dom.domain.repository.DomOrderRepository;
import com.metawebthree.dom.infrastructure.persistence.converter.DomOrderConverter;
import com.metawebthree.dom.infrastructure.persistence.dataobject.DomOrderDO;
import com.metawebthree.dom.infrastructure.persistence.mapper.DomOrderMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class DomOrderRepositoryImpl implements DomOrderRepository {

    private final DomOrderMapper domOrderMapper;
    private final DomOrderConverter domOrderConverter;

    public DomOrderRepositoryImpl(DomOrderMapper domOrderMapper, DomOrderConverter domOrderConverter) {
        this.domOrderMapper = domOrderMapper;
        this.domOrderConverter = domOrderConverter;
    }

    @Override
    public Optional<DomOrder> findById(Long id) {
        DomOrderDO domOrderDO = domOrderMapper.selectById(id);
        return Optional.ofNullable(domOrderConverter.toEntity(domOrderDO));
    }

    @Override
    public Optional<DomOrder> findByDomOrderNo(String domOrderNo) {
        LambdaQueryWrapper<DomOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DomOrderDO::getDomOrderNo, domOrderNo);
        DomOrderDO domOrderDO = domOrderMapper.selectOne(wrapper);
        return Optional.ofNullable(domOrderConverter.toEntity(domOrderDO));
    }

    @Override
    public Optional<DomOrder> findByOriginalOrderNo(String originalOrderNo) {
        LambdaQueryWrapper<DomOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DomOrderDO::getOriginalOrderNo, originalOrderNo);
        DomOrderDO domOrderDO = domOrderMapper.selectOne(wrapper);
        return Optional.ofNullable(domOrderConverter.toEntity(domOrderDO));
    }

    @Override
    public List<DomOrder> findByStatus(DomOrderStatus status) {
        LambdaQueryWrapper<DomOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DomOrderDO::getStatus, status.name());
        return domOrderMapper.selectList(wrapper).stream()
                .map(domOrderConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<DomOrder> findAll() {
        return domOrderMapper.selectList(null).stream()
                .map(domOrderConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public IPage<DomOrder> findPage(Page<DomOrder> page, DomOrderStatus status) {
        Page<DomOrderDO> doPage = new Page<>(page.getCurrent(), page.getSize());
        LambdaQueryWrapper<DomOrderDO> wrapper = new LambdaQueryWrapper<>();
        if (status != null) {
            wrapper.eq(DomOrderDO::getStatus, status.name());
        }
        wrapper.orderByDesc(DomOrderDO::getCreatedAt);
        IPage<DomOrderDO> doResult = domOrderMapper.selectPage(doPage, wrapper);
        IPage<DomOrder> result = new Page<>(doResult.getCurrent(), doResult.getSize(), doResult.getTotal());
        result.setRecords(doResult.getRecords().stream()
                .map(domOrderConverter::toEntity)
                .collect(Collectors.toList()));
        return result;
    }

    @Override
    public DomOrder save(DomOrder domOrder) {
        DomOrderDO domOrderDO = domOrderConverter.toDO(domOrder);
        if (domOrder.getId() == null) {
            domOrderMapper.insert(domOrderDO);
            domOrder.setId(domOrderDO.getId());
        } else {
            domOrderMapper.updateById(domOrderDO);
        }
        return domOrder;
    }
}
