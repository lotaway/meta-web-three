package com.metawebthree.finance.infrastructure.persistence.mapper.cash;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.finance.infrastructure.persistence.dataobject.cash.CashFlowForecastItemDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CashFlowForecastItemMapper extends BaseMapper<CashFlowForecastItemDO> {
}