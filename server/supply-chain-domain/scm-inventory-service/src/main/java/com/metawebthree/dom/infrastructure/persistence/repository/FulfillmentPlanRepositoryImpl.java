package com.metawebthree.dom.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.dom.domain.entity.FulfillmentPlan;
import com.metawebthree.dom.domain.repository.FulfillmentPlanRepository;
import com.metawebthree.dom.infrastructure.persistence.converter.FulfillmentPlanConverter;
import com.metawebthree.dom.infrastructure.persistence.dataobject.FulfillmentPlanDO;
import com.metawebthree.dom.infrastructure.persistence.mapper.FulfillmentPlanMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class FulfillmentPlanRepositoryImpl implements FulfillmentPlanRepository {

    private final FulfillmentPlanMapper fulfillmentPlanMapper;
    private final FulfillmentPlanConverter fulfillmentPlanConverter;

    public FulfillmentPlanRepositoryImpl(FulfillmentPlanMapper fulfillmentPlanMapper, FulfillmentPlanConverter fulfillmentPlanConverter) {
        this.fulfillmentPlanMapper = fulfillmentPlanMapper;
        this.fulfillmentPlanConverter = fulfillmentPlanConverter;
    }

    @Override
    public Optional<FulfillmentPlan> findById(Long id) {
        FulfillmentPlanDO doObj = fulfillmentPlanMapper.selectById(id);
        return Optional.ofNullable(fulfillmentPlanConverter.toEntity(doObj));
    }

    @Override
    public Optional<FulfillmentPlan> findByDomOrderId(Long domOrderId) {
        LambdaQueryWrapper<FulfillmentPlanDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(FulfillmentPlanDO::getDomOrderId, domOrderId);
        FulfillmentPlanDO doObj = fulfillmentPlanMapper.selectOne(wrapper);
        return Optional.ofNullable(fulfillmentPlanConverter.toEntity(doObj));
    }

    @Override
    public List<FulfillmentPlan> findByStatus(String status) {
        LambdaQueryWrapper<FulfillmentPlanDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(FulfillmentPlanDO::getStatus, status);
        return fulfillmentPlanMapper.selectList(wrapper).stream()
                .map(fulfillmentPlanConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public FulfillmentPlan save(FulfillmentPlan plan) {
        FulfillmentPlanDO doObj = fulfillmentPlanConverter.toDO(plan);
        if (plan.getId() == null) {
            fulfillmentPlanMapper.insert(doObj);
            plan.setId(doObj.getId());
        } else {
            fulfillmentPlanMapper.updateById(doObj);
        }
        return plan;
    }
}
