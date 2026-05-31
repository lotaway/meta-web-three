package com.metawebthree.order.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.order.domain.model.SagaInstance;
import org.apache.ibatis.annotations.Mapper;

/**
 * Saga instance mapper.
 */
@Mapper
public interface SagaInstanceMapper extends BaseMapper<SagaInstance> {
}
