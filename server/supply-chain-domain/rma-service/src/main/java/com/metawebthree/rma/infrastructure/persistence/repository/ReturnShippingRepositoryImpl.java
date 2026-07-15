package com.metawebthree.rma.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.rma.domain.entity.ReturnShipping;
import com.metawebthree.rma.domain.repository.ReturnShippingRepository;
import com.metawebthree.rma.infrastructure.persistence.converter.ReturnShippingConverter;
import com.metawebthree.rma.infrastructure.persistence.dataobject.ReturnShippingDO;
import com.metawebthree.rma.infrastructure.persistence.mapper.ReturnShippingMapper;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public class ReturnShippingRepositoryImpl implements ReturnShippingRepository {

    private final ReturnShippingMapper returnShippingMapper;
    private final ReturnShippingConverter returnShippingConverter;

    public ReturnShippingRepositoryImpl(ReturnShippingMapper returnShippingMapper,
                                        ReturnShippingConverter returnShippingConverter) {
        this.returnShippingMapper = returnShippingMapper;
        this.returnShippingConverter = returnShippingConverter;
    }

    @Override
    public Optional<ReturnShipping> findByRmaId(Long rmaId) {
        LambdaQueryWrapper<ReturnShippingDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReturnShippingDO::getRmaId, rmaId);
        ReturnShippingDO doObj = returnShippingMapper.selectOne(wrapper);
        return Optional.ofNullable(returnShippingConverter.toEntity(doObj));
    }

    @Override
    public Optional<ReturnShipping> findByTrackingNo(String trackingNo) {
        LambdaQueryWrapper<ReturnShippingDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReturnShippingDO::getTrackingNo, trackingNo);
        ReturnShippingDO doObj = returnShippingMapper.selectOne(wrapper);
        return Optional.ofNullable(returnShippingConverter.toEntity(doObj));
    }

    @Override
    public ReturnShipping save(ReturnShipping shipping) {
        ReturnShippingDO doObj = returnShippingConverter.toDO(shipping);
        if (shipping.getId() == null) {
            returnShippingMapper.insert(doObj);
            shipping.setId(doObj.getId());
        } else {
            returnShippingMapper.updateById(doObj);
        }
        return shipping;
    }
}
