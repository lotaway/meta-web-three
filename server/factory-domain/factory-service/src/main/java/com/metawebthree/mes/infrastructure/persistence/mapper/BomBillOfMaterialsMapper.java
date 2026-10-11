package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.BomBillOfMaterialsDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface BomBillOfMaterialsMapper extends BaseMapper<BomBillOfMaterialsDO> {
    
    BomBillOfMaterialsDO findByBomCode(@Param("bomCode") String bomCode);
    
    List<BomBillOfMaterialsDO> findByProductCode(@Param("productCode") String productCode);
    
    List<BomBillOfMaterialsDO> findActiveByProductCode(@Param("productCode") String productCode);
    
    List<BomBillOfMaterialsDO> findByProductCodeAndVersion(@Param("productCode") String productCode, @Param("version") String version);
}