package com.metawebthree.dataanalysis.infrastructure.persistence.mapper;

import com.metawebthree.dataanalysis.domain.entity.InventoryAnalysisDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface InventoryAnalysisMapper {
    void insert(InventoryAnalysisDO record);
    void updateById(InventoryAnalysisDO record);
    InventoryAnalysisDO selectByProductId(@Param("productId") String productId);
    List<InventoryAnalysisDO> selectAll();
    List<InventoryAnalysisDO> selectLowStock();
    List<InventoryAnalysisDO> selectOverstock();
    Long countLowStock();
    Long countOverstock();
}