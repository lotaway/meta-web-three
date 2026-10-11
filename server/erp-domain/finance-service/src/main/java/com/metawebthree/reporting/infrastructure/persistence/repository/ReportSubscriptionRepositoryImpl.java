package com.metawebthree.reporting.infrastructure.persistence.repository;

import com.metawebthree.reporting.domain.entity.ReportSubscription;
import com.metawebthree.reporting.domain.entity.ReportSubscription.ReportType;
import com.metawebthree.reporting.domain.repository.ReportSubscriptionRepository;
import com.metawebthree.reporting.infrastructure.persistence.converter.ReportSubscriptionConverter;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.ReportSubscriptionDO;
import com.metawebthree.reporting.infrastructure.persistence.mapper.ReportSubscriptionMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Repository
@RequiredArgsConstructor
public class ReportSubscriptionRepositoryImpl implements ReportSubscriptionRepository {

    private final ReportSubscriptionMapper subscriptionMapper;
    private final ReportSubscriptionConverter converter;

    @Override
    public Long save(ReportSubscription subscription) {
        subscription.setCreatedAt(LocalDateTime.now());
        subscription.setUpdatedAt(LocalDateTime.now());
        ReportSubscriptionDO dto = converter.toDO(subscription);
        subscriptionMapper.insert(dto);
        return dto.getId();
    }

    @Override
    public void update(ReportSubscription subscription) {
        subscription.setUpdatedAt(LocalDateTime.now());
        subscriptionMapper.updateById(converter.toDO(subscription));
    }

    @Override
    public ReportSubscription findById(Long id) {
        ReportSubscriptionDO dto = subscriptionMapper.selectById(id);
        return converter.toEntity(dto);
    }

    @Override
    public List<ReportSubscription> findByUserId(Long userId) {
        List<ReportSubscriptionDO> dtos = subscriptionMapper.selectByUserId(userId);
        return dtos.stream().map(converter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<ReportSubscription> findEnabled() {
        List<ReportSubscriptionDO> dtos = subscriptionMapper.selectEnabled();
        return dtos.stream().map(converter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<ReportSubscription> findDueSubscriptions(LocalDateTime currentTime) {
        List<ReportSubscriptionDO> dtos = subscriptionMapper.selectDueSubscriptions(currentTime);
        return dtos.stream().map(converter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<ReportSubscription> findByReportType(ReportType reportType) {
        List<ReportSubscriptionDO> dtos = subscriptionMapper.selectByReportType(reportType.name());
        return dtos.stream().map(converter::toEntity).collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        subscriptionMapper.deleteById(id);
    }
}