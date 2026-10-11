package com.metawebthree.warehouse.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.DefectRecordDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface DefectRecordMapper extends BaseMapper<DefectRecordDO> {
    
    @Select("SELECT * FROM defect_record WHERE inspection_id = #{inspectionId} AND deleted = 0")
    List<DefectRecordDO> selectByInspectionId(@Param("inspectionId") Long inspectionId);
}