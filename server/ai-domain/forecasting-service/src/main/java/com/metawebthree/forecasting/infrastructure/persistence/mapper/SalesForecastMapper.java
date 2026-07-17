package com.metawebthree.forecasting.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.forecasting.domain.entity.SalesForecast;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface SalesForecastMapper extends BaseMapper<SalesForecast> {
}
