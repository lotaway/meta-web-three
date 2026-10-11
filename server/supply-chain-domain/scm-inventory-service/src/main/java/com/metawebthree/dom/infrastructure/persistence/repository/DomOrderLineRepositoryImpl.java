package com.metawebthree.dom.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.dom.domain.entity.DomOrderLine;
import com.metawebthree.dom.domain.repository.DomOrderLineRepository;
import com.metawebthree.dom.infrastructure.persistence.converter.DomOrderLineConverter;
import com.metawebthree.dom.infrastructure.persistence.dataobject.DomOrderLineDO;
import com.metawebthree.dom.infrastructure.persistence.mapper.DomOrderLineMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.stream.Collectors;

@Repository
public class DomOrderLineRepositoryImpl implements DomOrderLineRepository {

    private final DomOrderLineMapper domOrderLineMapper;
    private final DomOrderLineConverter domOrderLineConverter;

    public DomOrderLineRepositoryImpl(DomOrderLineMapper domOrderLineMapper, DomOrderLineConverter domOrderLineConverter) {
        this.domOrderLineMapper = domOrderLineMapper;
        this.domOrderLineConverter = domOrderLineConverter;
    }

    @Override
    public List<DomOrderLine> findByDomOrderId(Long domOrderId) {
        LambdaQueryWrapper<DomOrderLineDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DomOrderLineDO::getDomOrderId, domOrderId);
        return domOrderLineMapper.selectList(wrapper).stream()
                .map(domOrderLineConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<DomOrderLine> findBySkuCode(String skuCode) {
        LambdaQueryWrapper<DomOrderLineDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DomOrderLineDO::getSkuCode, skuCode);
        return domOrderLineMapper.selectList(wrapper).stream()
                .map(domOrderLineConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public DomOrderLine save(DomOrderLine line) {
        DomOrderLineDO domOrderLineDO = domOrderLineConverter.toDO(line);
        if (line.getId() == null) {
            domOrderLineMapper.insert(domOrderLineDO);
            line.setId(domOrderLineDO.getId());
        } else {
            domOrderLineMapper.updateById(domOrderLineDO);
        }
        return line;
    }

    @Override
    public List<DomOrderLine> saveAll(List<DomOrderLine> lines) {
        return lines.stream()
                .map(this::save)
                .collect(Collectors.toList());
    }
}
