package com.metawebthree.procurement.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.procurement.infrastructure.persistence.dataobject.ProcurementOrderDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import java.util.List;

@Mapper
public interface ProcurementOrderMapper extends BaseMapper<ProcurementOrderDO> {
    
    @Select("SELECT * FROM procurement_order WHERE order_no = #{orderNo}")
    ProcurementOrderDO selectByOrderNo(@Param("orderNo") String orderNo);
    
    @Select("SELECT * FROM procurement_order WHERE status = #{status}")
    List<ProcurementOrderDO> selectByStatus(@Param("status") String status);
    
    @Select("SELECT * FROM procurement_order WHERE supplier_code = #{supplierCode}")
    List<ProcurementOrderDO> selectBySupplierCode(@Param("supplierCode") String supplierCode);
    
    @Select("SELECT * FROM procurement_order WHERE warehouse_id = #{warehouseId}")
    List<ProcurementOrderDO> selectByWarehouseId(@Param("warehouseId") Long warehouseId);
}