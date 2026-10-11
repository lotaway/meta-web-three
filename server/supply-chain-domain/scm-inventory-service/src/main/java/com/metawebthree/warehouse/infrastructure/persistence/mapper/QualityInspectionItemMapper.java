package com.metawebthree.warehouse.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.QualityInspectionItemDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface QualityInspectionItemMapper extends BaseMapper<QualityInspectionItemDO> {
    
    @Select("SELECT * FROM quality_inspection_item WHERE inspection_id = #{inspectionId} AND deleted = 0")
    List<QualityInspectionItemDO> selectByInspectionId(@Param("inspectionId") Long inspectionId);
}