package com.metawebthree.reporting.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.FinancialReportDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface FinancialReportMapper extends BaseMapper<FinancialReportDO> {
}