package com.metawebthree.developerportal.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.developerportal.entity.ApiUsageStats;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface ApiUsageStatsRepository extends BaseMapper<ApiUsageStats> {

    default List<ApiUsageStats> findByDeveloperIdAndStatTimeBetween(
            String developerId, LocalDateTime startTime, LocalDateTime endTime) {
        return selectList(new QueryWrapper<ApiUsageStats>()
            .eq("developer_id", developerId)
            .between("stat_time", startTime, endTime));
    }

    default List<ApiUsageStats> findByDeveloperIdAndApiEndpointAndStatTimeBetween(
            String developerId, String apiEndpoint,
            LocalDateTime startTime, LocalDateTime endTime) {
        return selectList(new QueryWrapper<ApiUsageStats>()
            .eq("developer_id", developerId)
            .eq("api_endpoint", apiEndpoint)
            .between("stat_time", startTime, endTime));
    }

    @Select("SELECT COALESCE(SUM(request_count), 0) FROM api_usage_stats " +
            "WHERE developer_id = #{developerId} AND stat_time BETWEEN #{startTime} AND #{endTime}")
    Long sumRequestCountByDeveloperAndTimeRange(
            @Param("developerId") String developerId,
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime);

    @Select("SELECT COALESCE(SUM(billing_amount_cents), 0) FROM api_usage_stats " +
            "WHERE developer_id = #{developerId} AND stat_time BETWEEN #{startTime} AND #{endTime}")
    Long sumBillingAmountByDeveloperAndTimeRange(
            @Param("developerId") String developerId,
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime);

    default List<ApiUsageStats> findByDeveloperIdAndStatTimeBetweenOrderByStatTimeDesc(
            String developerId, LocalDateTime startTime, LocalDateTime endTime) {
        return selectList(new QueryWrapper<ApiUsageStats>()
            .eq("developer_id", developerId)
            .between("stat_time", startTime, endTime)
            .orderByDesc("stat_time"));
    }

    @Select("SELECT api_endpoint, SUM(request_count) as total FROM api_usage_stats " +
            "WHERE developer_id = #{developerId} AND stat_time BETWEEN #{startTime} AND #{endTime} " +
            "GROUP BY api_endpoint ORDER BY total DESC")
    List<Object[]> findTopEndpointsByDeveloper(
            @Param("developerId") String developerId,
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime);

    default void save(ApiUsageStats entity) {
        if (entity.getId() == null) {
            insert(entity);
        } else {
            updateById(entity);
        }
    }
}
