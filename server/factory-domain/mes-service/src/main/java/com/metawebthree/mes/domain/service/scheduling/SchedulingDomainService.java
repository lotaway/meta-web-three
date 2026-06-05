package com.metawebthree.mes.domain.service.scheduling;

import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResult;

import java.util.List;

public interface SchedulingDomainService {
    ScheduleResult forwardSchedule(List<ScheduleOrder> orders, String workshopId);
    ScheduleResult backwardSchedule(List<ScheduleOrder> orders, String workshopId);
    ScheduleResult reschedule(Long orderId);
    void releaseResource(Long resourceId, java.time.LocalDateTime endTime);
}
