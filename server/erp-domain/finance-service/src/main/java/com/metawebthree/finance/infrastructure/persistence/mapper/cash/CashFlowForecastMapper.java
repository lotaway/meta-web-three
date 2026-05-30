package com.metawebthree.finance.infrastructure.persistence.mapper.cash;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.finance.infrastructure.persistence.dataobject.cash.CashFlowForecastDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CashFlowForecastMapper extends BaseMapper<CashFlowForecastDO> {
}