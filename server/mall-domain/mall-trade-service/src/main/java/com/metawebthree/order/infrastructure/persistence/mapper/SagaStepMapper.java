package com.metawebthree.order.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.order.domain.model.SagaStep;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

/**
 * Saga step mapper.
 */
@Mapper
public interface SagaStepMapper extends BaseMapper<SagaStep> {
    
    /**
     * Get steps by saga ID ordered by step order.
     */
    default List<SagaStep> selectBySagaIdOrdered(String sagaId) {
        return selectList(
            new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<SagaStep>()
                .eq(SagaStep::getSagaId, sagaId)
                .orderByAsc(SagaStep::getStepOrder)
        );
    }
}
