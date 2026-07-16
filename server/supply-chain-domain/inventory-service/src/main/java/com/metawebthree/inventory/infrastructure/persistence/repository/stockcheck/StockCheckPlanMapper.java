package com.metawebthree.inventory.infrastructure.persistence.repository.stockcheck;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckPlan;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface StockCheckPlanMapper extends BaseMapper<StockCheckPlan> {
}
