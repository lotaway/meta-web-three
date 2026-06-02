package com.metawebthree.dataanalysis.infrastructure.persistence.mapper;

import com.metawebthree.dataanalysis.domain.entity.ProductSalesDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface ProductSalesMapper {
    void insert(ProductSalesDO record);
    void updateById(ProductSalesDO record);
    ProductSalesDO selectByDateAndProductId(@Param("date") String date, @Param("productId") Long productId);
    List<ProductSalesDO> selectByDate(@Param("date") String date);
    List<ProductSalesDO> selectTopProducts(@Param("date") String date, @Param("limit") Integer limit);
    List<ProductSalesDO> selectByDateRange(@Param("startDate") String startDate, @Param("endDate") String endDate);
    List<ProductSalesDO> selectAll();
}