package com.metawebthree.finance.infrastructure.persistence.mapper.cash;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.finance.infrastructure.persistence.dataobject.cash.BankReconciliationDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface BankReconciliationMapper extends BaseMapper<BankReconciliationDO> {
}