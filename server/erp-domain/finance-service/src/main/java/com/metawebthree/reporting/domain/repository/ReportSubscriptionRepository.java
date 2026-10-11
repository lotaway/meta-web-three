package com.metawebthree.reporting.domain.repository;

import com.metawebthree.reporting.domain.entity.ReportSubscription;
import com.metawebthree.reporting.domain.entity.ReportSubscription.ReportType;

import java.time.LocalDateTime;
import java.util.List;

public interface ReportSubscriptionRepository {

    Long save(ReportSubscription subscription);

    void update(ReportSubscription subscription);

    ReportSubscription findById(Long id);

    List<ReportSubscription> findByUserId(Long userId);

    List<ReportSubscription> findEnabled();

    List<ReportSubscription> findDueSubscriptions(LocalDateTime currentTime);

    List<ReportSubscription> findByReportType(ReportType reportType);

    void delete(Long id);
}