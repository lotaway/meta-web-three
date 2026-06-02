package com.metawebthree.finance.infrastructure.persistence.mapper.arap;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.finance.infrastructure.persistence.dataobject.arap.AccountsPayableDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import java.time.LocalDate;
import java.util.List;

@Mapper
public interface AccountsPayableMapper extends BaseMapper<AccountsPayableDO> {
    
    @Select("SELECT * FROM accounts_payable WHERE supplier_id = #{supplierId}")
    default List<AccountsPayableDO> findBySupplierId(Long supplierId) {
        return selectList(new LambdaQueryWrapper<AccountsPayableDO>()
            .eq(AccountsPayableDO::getSupplierId, supplierId));
    }
    
    @Select("SELECT * FROM accounts_payable WHERE status = #{status}")
    default List<AccountsPayableDO> findByStatus(String status) {
        return selectList(new LambdaQueryWrapper<AccountsPayableDO>()
            .eq(AccountsPayableDO::getStatus, status));
    }
    
    @Select("SELECT * FROM accounts_payable WHERE due_date < #{date}")
    default List<AccountsPayableDO> findByDueDateBefore(LocalDate date) {
        return selectList(new LambdaQueryWrapper<AccountsPayableDO>()
            .lt(AccountsPayableDO::getDueDate, date));
    }
    
    @Select("SELECT * FROM accounts_payable WHERE supplier_id = #{supplierId} AND status = #{status}")
    default List<AccountsPayableDO> findBySupplierIdAndStatus(Long supplierId, String status) {
        return selectList(new LambdaQueryWrapper<AccountsPayableDO>()
            .eq(AccountsPayableDO::getSupplierId, supplierId)
            .eq(AccountsPayableDO::getStatus, status));
    }
    
    @Select("SELECT * FROM accounts_payable WHERE ap_code = #{apCode} LIMIT 1")
    default AccountsPayableDO findByApCode(String apCode) {
        return selectOne(new LambdaQueryWrapper<AccountsPayableDO>()
            .eq(AccountsPayableDO::getApCode, apCode));
    }
}