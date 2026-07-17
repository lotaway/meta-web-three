package com.metawebthree.routeoptimizer.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.routeoptimizer.domain.entity.RoutePoint;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface RoutePointMapper extends BaseMapper<RoutePoint> {
}
