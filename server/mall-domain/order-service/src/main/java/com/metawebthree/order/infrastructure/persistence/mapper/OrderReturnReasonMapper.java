package com.metawebthree.order.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.order.domain.model.OrderReturnReasonDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface OrderReturnReasonMapper extends BaseMapper<OrderReturnReasonDO> {
}
