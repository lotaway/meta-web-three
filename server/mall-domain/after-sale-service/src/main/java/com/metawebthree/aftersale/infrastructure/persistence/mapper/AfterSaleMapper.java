package com.metawebthree.aftersale.infrastructure.persistence.mapper;

import com.metawebthree.aftersale.domain.model.AfterSaleOrderDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface AfterSaleMapper {
    int insert(AfterSaleOrderDO record);
    int update(AfterSaleOrderDO record);
    int deleteById(Long id);
    AfterSaleOrderDO selectById(Long id);
    List<AfterSaleOrderDO> selectByUserId(Long userId);
    List<AfterSaleOrderDO> selectByOrderId(Long orderId);
    List<AfterSaleOrderDO> selectAll();
    int updateStatus(@Param("id") Long id, @Param("status") Integer status);
}