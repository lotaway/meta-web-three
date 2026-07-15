package com.metawebthree.rma.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.rma.domain.entity.RmaDisposition;
import com.metawebthree.rma.domain.repository.RmaDispositionRepository;
import com.metawebthree.rma.infrastructure.persistence.converter.RmaDispositionConverter;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaDispositionDO;
import com.metawebthree.rma.infrastructure.persistence.mapper.RmaDispositionMapper;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public class RmaDispositionRepositoryImpl implements RmaDispositionRepository {

    private final RmaDispositionMapper rmaDispositionMapper;
    private final RmaDispositionConverter rmaDispositionConverter;

    public RmaDispositionRepositoryImpl(RmaDispositionMapper rmaDispositionMapper,
                                        RmaDispositionConverter rmaDispositionConverter) {
        this.rmaDispositionMapper = rmaDispositionMapper;
        this.rmaDispositionConverter = rmaDispositionConverter;
    }

    @Override
    public Optional<RmaDisposition> findByRmaId(Long rmaId) {
        LambdaQueryWrapper<RmaDispositionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RmaDispositionDO::getRmaId, rmaId);
        RmaDispositionDO doObj = rmaDispositionMapper.selectOne(wrapper);
        return Optional.ofNullable(rmaDispositionConverter.toEntity(doObj));
    }

    @Override
    public RmaDisposition save(RmaDisposition disposition) {
        RmaDispositionDO doObj = rmaDispositionConverter.toDO(disposition);
        if (disposition.getId() == null) {
            rmaDispositionMapper.insert(doObj);
            disposition.setId(doObj.getId());
        } else {
            rmaDispositionMapper.updateById(doObj);
        }
        return disposition;
    }
}
