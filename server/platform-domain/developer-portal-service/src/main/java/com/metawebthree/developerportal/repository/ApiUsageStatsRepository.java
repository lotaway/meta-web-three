package com.metawebthree.developerportal.repository;

import com.metawebthree.developerportal.entity.ApiUsageStats;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

/**
 * API Usage Statistics Repository
 */
@Repository
public interface ApiUsageStatsRepository extends JpaRepository<ApiUsageStats, Long> {

    /**
     * Find usage stats by developer ID and time range
     */
    List<ApiUsageStats> findByDeveloperIdAndStatTimeBetween(
        String developerId, 
        LocalDateTime startTime, 
        LocalDateTime endTime
    );

    /**
     * Find usage stats by developer ID, API endpoint and time range
     */
    List<ApiUsageStats> findByDeveloperIdAndApiEndpointAndStatTimeBetween(
        String developerId,
        String apiEndpoint,
        LocalDateTime startTime,
        LocalDateTime endTime
    );

    /**
     * Sum total requests by developer in time range
     */
    @Query("SELECT SUM(s.requestCount) FROM ApiUsageStats s WHERE s.developerId = :developerId AND s.statTime BETWEEN :startTime AND :endTime")
    Long sumRequestCountByDeveloperAndTimeRange(
        @Param("developerId") String developerId,
        @Param("startTime") LocalDateTime startTime,
        @Param("endTime") LocalDateTime endTime
    );

    /**
     * Sum billing amount by developer in time range
     */
    @Query("SELECT SUM(s.billingAmountCents) FROM ApiUsageStats s WHERE s.developerId = :developerId AND s.statTime BETWEEN :startTime AND :endTime")
    Long sumBillingAmountByDeveloperAndTimeRange(
        @Param("developerId") String developerId,
        @Param("startTime") LocalDateTime startTime,
        @Param("endTime") LocalDateTime endTime
    );

    /**
     * Find usage stats by developer ID and time range, ordered by stat time descending
     */
    List<ApiUsageStats> findByDeveloperIdAndStatTimeBetweenOrderByStatTimeDesc(
        String developerId,
        LocalDateTime startTime,
        LocalDateTime endTime
    );

    /**
     * Get top API endpoints by request count for a developer
     */
    @Query("SELECT s.apiEndpoint, SUM(s.requestCount) as total FROM ApiUsageStats s WHERE s.developerId = :developerId AND s.statTime BETWEEN :startTime AND :endTime GROUP BY s.apiEndpoint ORDER BY total DESC")
    List<Object[]> findTopEndpointsByDeveloper(
        @Param("developerId") String developerId,
        @Param("startTime") LocalDateTime startTime,
        @Param("endTime") LocalDateTime endTime
    );
}
