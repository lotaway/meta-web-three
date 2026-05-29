package com.metawebthree.supplier.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import java.util.List;

@Mapper
public interface SupplierMapper extends BaseMapper<SupplierDO> {
    
    @Select("SELECT * FROM supplier WHERE supplier_code = #{code}")
    SupplierDO selectByCode(@Param("code") String code);
    
    @Select("SELECT * FROM supplier WHERE status = #{status}")
    List<SupplierDO> selectByStatus(@Param("status") String status);
    
    @Select("SELECT * FROM supplier WHERE category = #{category}")
    List<SupplierDO> selectByCategory(@Param("category") String category);
    
    @Select("SELECT * FROM supplier WHERE assessment_level = #{level}")
    List<SupplierDO> selectByAssessmentLevel(@Param("level") String level);
}