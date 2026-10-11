package com.metawebthree.dataanalysis.infrastructure.persistence.mapper;

import com.metawebthree.dataanalysis.domain.entity.HourlySalesDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface HourlySalesMapper {
    void insert(HourlySalesDO record);
    void updateById(HourlySalesDO record);
    HourlySalesDO selectByDateAndHour(@Param("date") String date, @Param("hour") Integer hour);
    List<HourlySalesDO> selectByDate(@Param("date") String date);
    List<HourlySalesDO> selectByDateRange(@Param("startDate") String startDate, @Param("endDate") String endDate);
    List<HourlySalesDO> selectAll();
}