package com.metawebthree.riskcontrol.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.riskcontrol.domain.RiskEvent;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface RiskEventRepository extends BaseMapper<RiskEvent> {
}