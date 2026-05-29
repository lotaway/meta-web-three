package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialSubstituteDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface MaterialSubstituteMapper extends BaseMapper<MaterialSubstituteDO> {
    
    MaterialSubstituteDO findByProductCodeAndMainMaterialCode(@Param("productCode") String productCode, 
                                                               @Param("mainMaterialCode") String mainMaterialCode);
    
    List<MaterialSubstituteDO> findByProductCode(@Param("productCode") String productCode);
    
    List<MaterialSubstituteDO> findActiveByProductCode(@Param("productCode") String productCode);
}