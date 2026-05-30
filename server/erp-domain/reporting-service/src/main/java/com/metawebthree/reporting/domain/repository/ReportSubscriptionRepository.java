package com.metawebthree.reporting.domain.repository;

import com.metawebthree.reporting.domain.entity.ReportSubscription;
import com.metawebthree.reporting.domain.entity.ReportSubscription.ReportType;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 报表订阅仓储接口
 */
public interface ReportSubscriptionRepository {

    /**
     * 保存订阅
     */
    Long save(ReportSubscription subscription);

    /**
     * 更新订阅
     */
    void update(ReportSubscription subscription);

    /**
     * 根据ID查询订阅
     */
    ReportSubscription findById(Long id);

    /**
     * 根据用户ID查询订阅列表
     */
    List<ReportSubscription> findByUserId(Long userId);

    /**
     * 查询所有启用的订阅
     */
    List<ReportSubscription> findEnabled();

    /**
     * 查询需要发送的订阅（下次发送时间已过或已到）
     */
    List<ReportSubscription> findDueSubscriptions(LocalDateTime currentTime);

    /**
     * 根据报表类型查询订阅列表
     */
    List<ReportSubscription> findByReportType(ReportType reportType);

    /**
     * 删除订阅
     */
    void delete(Long id);
}