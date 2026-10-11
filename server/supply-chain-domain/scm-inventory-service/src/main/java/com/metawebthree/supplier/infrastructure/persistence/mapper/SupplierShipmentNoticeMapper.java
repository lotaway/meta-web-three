package com.metawebthree.supplier.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierShipmentNoticeDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import java.util.List;

@Mapper
public interface SupplierShipmentNoticeMapper extends BaseMapper<SupplierShipmentNoticeDO> {
    
    @Select("SELECT * FROM supplier_shipment_notice WHERE notice_no = #{noticeNo}")
    SupplierShipmentNoticeDO selectByNoticeNo(@Param("noticeNo") String noticeNo);
    
    @Select("SELECT * FROM supplier_shipment_notice WHERE supplier_code = #{supplierCode}")
    List<SupplierShipmentNoticeDO> selectBySupplierCode(@Param("supplierCode") String supplierCode);
    
    @Select("SELECT * FROM supplier_shipment_notice WHERE order_no = #{orderNo}")
    List<SupplierShipmentNoticeDO> selectByOrderNo(@Param("orderNo") String orderNo);
    
    @Select("SELECT * FROM supplier_shipment_notice WHERE supplier_code = #{supplierCode} AND status = #{status}")
    List<SupplierShipmentNoticeDO> selectBySupplierCodeAndStatus(@Param("supplierCode") String supplierCode, @Param("status") String status);
}