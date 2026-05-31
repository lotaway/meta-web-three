package com.metawebthree.dataanalysis.infrastructure.persistence.mapper;

import com.metawebthree.dataanalysis.domain.entity.SalesStatisticsDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface SalesStatisticsMapper {
    void insert(SalesStatisticsDO record);
    void updateById(SalesStatisticsDO record);
    SalesStatisticsDO selectByDate(@Param("date") String date);
    List<SalesStatisticsDO> selectByDateRange(@Param("startDate") String startDate, @Param("endDate") String endDate);
    List<SalesStatisticsDO> selectAll();
}