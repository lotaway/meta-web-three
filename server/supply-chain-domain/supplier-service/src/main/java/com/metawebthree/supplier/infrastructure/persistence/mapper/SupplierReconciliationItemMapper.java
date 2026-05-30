package com.metawebthree.supplier.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierReconciliationItemDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import java.util.List;

@Mapper
public interface SupplierReconciliationItemMapper extends BaseMapper<SupplierReconciliationItemDO> {
    
    @Select("SELECT * FROM supplier_reconciliation_item WHERE reconciliation_id = #{reconciliationId}")
    List<SupplierReconciliationItemDO> selectByReconciliationId(@Param("reconciliationId") Long reconciliationId);
}