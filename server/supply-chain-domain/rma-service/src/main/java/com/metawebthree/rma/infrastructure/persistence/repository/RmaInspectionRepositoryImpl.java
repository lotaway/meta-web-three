package com.metawebthree.rma.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.rma.domain.entity.RmaInspection;
import com.metawebthree.rma.domain.repository.RmaInspectionRepository;
import com.metawebthree.rma.infrastructure.persistence.converter.RmaInspectionConverter;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaInspectionDO;
import com.metawebthree.rma.infrastructure.persistence.mapper.RmaInspectionMapper;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public class RmaInspectionRepositoryImpl implements RmaInspectionRepository {

    private final RmaInspectionMapper rmaInspectionMapper;
    private final RmaInspectionConverter rmaInspectionConverter;

    public RmaInspectionRepositoryImpl(RmaInspectionMapper rmaInspectionMapper,
                                       RmaInspectionConverter rmaInspectionConverter) {
        this.rmaInspectionMapper = rmaInspectionMapper;
        this.rmaInspectionConverter = rmaInspectionConverter;
    }

    @Override
    public Optional<RmaInspection> findByRmaId(Long rmaId) {
        LambdaQueryWrapper<RmaInspectionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RmaInspectionDO::getRmaId, rmaId);
        RmaInspectionDO doObj = rmaInspectionMapper.selectOne(wrapper);
        return Optional.ofNullable(rmaInspectionConverter.toEntity(doObj));
    }

    @Override
    public RmaInspection save(RmaInspection inspection) {
        RmaInspectionDO doObj = rmaInspectionConverter.toDO(inspection);
        if (inspection.getId() == null) {
            rmaInspectionMapper.insert(doObj);
            inspection.setId(doObj.getId());
        } else {
            rmaInspectionMapper.updateById(doObj);
        }
        return inspection;
    }
}
