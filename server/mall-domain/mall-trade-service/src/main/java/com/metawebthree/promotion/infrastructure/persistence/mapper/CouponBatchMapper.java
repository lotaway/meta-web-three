package com.metawebthree.promotion.infrastructure.persistence.mapper;

import org.apache.ibatis.annotations.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.CouponBatchRecord;

@Mapper
public interface CouponBatchMapper extends BaseMapper<CouponBatchRecord> {
}
