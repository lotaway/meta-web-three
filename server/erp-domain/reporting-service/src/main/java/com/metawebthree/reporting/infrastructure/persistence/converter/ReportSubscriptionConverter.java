package com.metawebthree.reporting.infrastructure.persistence.converter;

import com.metawebthree.reporting.domain.entity.ReportSubscription;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.ReportSubscriptionDO;
import org.springframework.stereotype.Component;

/**
 * 报表订阅转换器
 */
@Component
public class ReportSubscriptionConverter {

    public ReportSubscription toEntity(ReportSubscriptionDO dto) {
        if (dto == null) {
            return null;
        }
        ReportSubscription entity = new ReportSubscription();
        entity.setId(dto.getId());
        entity.setUserId(dto.getUserId());
        entity.setUserName(dto.getUserName());
        entity.setReportType(ReportSubscription.ReportType.valueOf(dto.getReportType()));
        entity.setFrequency(ReportSubscription.Frequency.valueOf(dto.getFrequency()));
        entity.setCronExpression(dto.getCronExpression());
        entity.setChannel(ReportSubscription.Channel.valueOf(dto.getChannel()));
        entity.setRecipient(dto.getRecipient());
        entity.setWebhookUrl(dto.getWebhookUrl());
        entity.setEnabled(dto.getEnabled());
        entity.setNextSendTime(dto.getNextSendTime());
        entity.setLastSendTime(dto.getLastSendTime());
        entity.setCreatedAt(dto.getCreatedAt());
        entity.setUpdatedAt(dto.getUpdatedAt());
        return entity;
    }

    public ReportSubscriptionDO toDO(ReportSubscription entity) {
        if (entity == null) {
            return null;
        }
        ReportSubscriptionDO dto = new ReportSubscriptionDO();
        dto.setId(entity.getId());
        dto.setUserId(entity.getUserId());
        dto.setUserName(entity.getUserName());
        dto.setReportType(entity.getReportType().name());
        dto.setFrequency(entity.getFrequency().name());
        dto.setCronExpression(entity.getCronExpression());
        dto.setChannel(entity.getChannel().name());
        dto.setRecipient(entity.getRecipient());
        dto.setWebhookUrl(entity.getWebhookUrl());
        dto.setEnabled(entity.getEnabled());
        dto.setNextSendTime(entity.getNextSendTime());
        dto.setLastSendTime(entity.getLastSendTime());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        return dto;
    }
}