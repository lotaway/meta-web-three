package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProductSnRuleDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 产品SN规则绑定 Mapper
 */
@Mapper
public interface ProductSnRuleMapper extends BaseMapper<ProductSnRuleDO> {
    
    @Select("SELECT * FROM mes_product_sn_rule WHERE product_id = #{productId} AND is_active = true LIMIT 1")
    ProductSnRuleDO findActiveByProductId(@Param("productId") Long productId);
    
    @Select("SELECT * FROM mes_product_sn_rule WHERE product_code = #{productCode} AND is_active = true LIMIT 1")
    ProductSnRuleDO findActiveByProductCode(@Param("productCode") String productCode);
    
    @Select("SELECT * FROM mes_product_sn_rule WHERE is_active = true")
    List<ProductSnRuleDO> findAllActive();
}