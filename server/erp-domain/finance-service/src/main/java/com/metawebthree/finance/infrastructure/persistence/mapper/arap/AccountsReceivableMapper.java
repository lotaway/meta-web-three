package com.metawebthree.finance.infrastructure.persistence.mapper.arap;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.finance.infrastructure.persistence.dataobject.arap.AccountsReceivableDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import java.time.LocalDate;
import java.util.List;

@Mapper
public interface AccountsReceivableMapper extends BaseMapper<AccountsReceivableDO> {
    
    @Select("SELECT * FROM accounts_receivable WHERE customer_id = #{customerId}")
    default List<AccountsReceivableDO> findByCustomerId(Long customerId) {
        return selectList(new LambdaQueryWrapper<AccountsReceivableDO>()
            .eq(AccountsReceivableDO::getCustomerId, customerId));
    }
    
    @Select("SELECT * FROM accounts_receivable WHERE status = #{status}")
    default List<AccountsReceivableDO> findByStatus(String status) {
        return selectList(new LambdaQueryWrapper<AccountsReceivableDO>()
            .eq(AccountsReceivableDO::getStatus, status));
    }
    
    @Select("SELECT * FROM accounts_receivable WHERE due_date < #{date}")
    default List<AccountsReceivableDO> findByDueDateBefore(LocalDate date) {
        return selectList(new LambdaQueryWrapper<AccountsReceivableDO>()
            .lt(AccountsReceivableDO::getDueDate, date));
    }
    
    @Select("SELECT * FROM accounts_receivable WHERE customer_id = #{customerId} AND status = #{status}")
    default List<AccountsReceivableDO> findByCustomerIdAndStatus(Long customerId, String status) {
        return selectList(new LambdaQueryWrapper<AccountsReceivableDO>()
            .eq(AccountsReceivableDO::getCustomerId, customerId)
            .eq(AccountsReceivableDO::getStatus, status));
    }
    
    @Select("SELECT * FROM accounts_receivable WHERE ar_code = #{arCode} LIMIT 1")
    default AccountsReceivableDO findByArCode(String arCode) {
        return selectOne(new LambdaQueryWrapper<AccountsReceivableDO>()
            .eq(AccountsReceivableDO::getArCode, arCode));
    }
}