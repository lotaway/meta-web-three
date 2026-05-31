package com.metawebthree.payment.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.payment.infrastructure.persistence.dataobject.ReconciliationDiffDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDate;
import java.util.List;

@Mapper
public interface ReconciliationDiffMapper extends BaseMapper<ReconciliationDiffDO> {

    @Select("SELECT * FROM payment_reconciliation_diff WHERE reconciliation_date = #{reconciliationDate}")
    List<ReconciliationDiffDO> findByReconciliationDate(@Param("reconciliationDate") LocalDate reconciliationDate);

    @Select("SELECT * FROM payment_reconciliation_diff WHERE reconciliation_date = #{reconciliationDate} AND status = #{status}")
    List<ReconciliationDiffDO> findByReconciliationDateAndStatus(@Param("reconciliationDate") LocalDate reconciliationDate,
                                                                  @Param("status") String status);

    @Select("SELECT COUNT(*) FROM payment_reconciliation_diff WHERE reconciliation_date = #{reconciliationDate} AND diff_type = #{diffType}")
    Long countByReconciliationDateAndDiffType(@Param("reconciliationDate") LocalDate reconciliationDate,
                                               @Param("diffType") String diffType);
}