package com.metawebthree.warehouse.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.DefectProcessingDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface DefectProcessingMapper extends BaseMapper<DefectProcessingDO> {
    
    @Select("SELECT * FROM defect_processing WHERE defect_id = #{defectId} AND deleted = 0")
    List<DefectProcessingDO> selectByDefectId(@Param("defectId") Long defectId);
    
    @Select("SELECT * FROM defect_processing WHERE processing_no = #{processingNo} AND deleted = 0")
    DefectProcessingDO selectByProcessingNo(@Param("processingNo") String processingNo);
}