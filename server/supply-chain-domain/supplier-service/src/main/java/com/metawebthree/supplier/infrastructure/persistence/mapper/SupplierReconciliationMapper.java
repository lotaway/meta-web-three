package com.metawebthree.supplier.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierReconciliationDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import java.time.LocalDate;
import java.util.List;

@Mapper
public interface SupplierReconciliationMapper extends BaseMapper<SupplierReconciliationDO> {
    
    @Select("SELECT * FROM supplier_reconciliation WHERE reconciliation_no = #{reconciliationNo}")
    SupplierReconciliationDO selectByReconciliationNo(@Param("reconciliationNo") String reconciliationNo);
    
    @Select("SELECT * FROM supplier_reconciliation WHERE supplier_code = #{supplierCode}")
    List<SupplierReconciliationDO> selectBySupplierCode(@Param("supplierCode") String supplierCode);
    
    @Select("SELECT * FROM supplier_reconciliation WHERE supplier_code = #{supplierCode} AND status = #{status}")
    List<SupplierReconciliationDO> selectBySupplierCodeAndStatus(@Param("supplierCode") String supplierCode, @Param("status") String status);
    
    @Select("SELECT * FROM supplier_reconciliation WHERE supplier_code = #{supplierCode} AND period_start >= #{periodStart} AND period_end <= #{periodEnd}")
    List<SupplierReconciliationDO> selectByPeriod(@Param("supplierCode") String supplierCode, @Param("periodStart") LocalDate periodStart, @Param("periodEnd") LocalDate periodEnd);
    
    @Select("SELECT * FROM supplier_reconciliation WHERE status = #{status}")
    List<SupplierReconciliationDO> selectByStatus(@Param("status") String status);
}