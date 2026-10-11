package com.metawebthree.supplier.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierShipmentNoticeItemDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import java.util.List;

@Mapper
public interface SupplierShipmentNoticeItemMapper extends BaseMapper<SupplierShipmentNoticeItemDO> {
    
    @Select("SELECT * FROM supplier_shipment_notice_item WHERE notice_id = #{noticeId}")
    List<SupplierShipmentNoticeItemDO> selectByNoticeId(@Param("noticeId") Long noticeId);
}