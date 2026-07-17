package com.metawebthree.forecasting.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.forecasting.domain.entity.ForecastModel;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ForecastModelMapper extends BaseMapper<ForecastModel> {
}
