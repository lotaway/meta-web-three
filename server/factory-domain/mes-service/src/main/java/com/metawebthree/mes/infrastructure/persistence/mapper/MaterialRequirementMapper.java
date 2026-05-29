package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialRequirementDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface MaterialRequirementMapper extends BaseMapper<MaterialRequirementDO> {
    
    MaterialRequirementDO findByRequirementNo(@Param("requirementNo") String requirementNo);
    
    MaterialRequirementDO findByWorkOrderNo(@Param("workOrderNo") String workOrderNo);
    
    List<MaterialRequirementDO> findByStatus(@Param("status") String status);
    
    List<MaterialRequirementDO> findByWarehouseId(@Param("warehouseId") String warehouseId);
    
    List<MaterialRequirementDO> findByWorkshopId(@Param("workshopId") String workshopId);
    
    List<MaterialRequirementDO> findByProductCode(@Param("productCode") String productCode);
}