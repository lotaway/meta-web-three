package com.metawebthree.warehouse.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.QualityInspectionDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface QualityInspectionMapper extends BaseMapper<QualityInspectionDO> {
    
    @Select("SELECT * FROM quality_inspection WHERE inspection_no = #{inspectionNo} AND deleted = 0")
    QualityInspectionDO selectByInspectionNo(@Param("inspectionNo") String inspectionNo);
    
    @Select("SELECT * FROM quality_inspection WHERE order_id = #{orderId} AND deleted = 0")
    List<QualityInspectionDO> selectByOrderId(@Param("orderId") Long orderId);
}