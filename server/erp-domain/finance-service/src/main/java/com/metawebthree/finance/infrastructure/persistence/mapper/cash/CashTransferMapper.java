package com.metawebthree.finance.infrastructure.persistence.mapper.cash;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.finance.infrastructure.persistence.dataobject.cash.CashTransferDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface CashTransferMapper extends BaseMapper<CashTransferDO> {
    
    @Select("SELECT * FROM cash_transfer WHERE from_account_id = #{accountId} OR to_account_id = #{accountId}")
    List<CashTransferDO> findByFromAccountIdOrToAccountId(@Param("accountId") Long accountId);
}