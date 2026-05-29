package com.metawebthree.logistics.infrastructure.persistence.converter;

import com.metawebthree.logistics.domain.entity.TrackingEvent;
import com.metawebthree.logistics.infrastructure.persistence.dataobject.TrackingEventDO;
import org.springframework.stereotype.Component;

@Component
public class TrackingEventConverter {

    public TrackingEvent toEntity(TrackingEventDO doObj) {
        if (doObj == null) {
            return null;
        }
        TrackingEvent entity = new TrackingEvent();
        entity.setId(doObj.getId());
        entity.setTrackingNo(doObj.getTrackingNo());
        entity.setEventType(doObj.getEventType());
        entity.setLocation(doObj.getLocation());
        entity.setDescription(doObj.getDescription());
        entity.setOperator(doObj.getOperator());
        entity.setOccurredAt(doObj.getOccurredAt());
        entity.setCreatedAt(doObj.getCreatedAt());
        return entity;
    }

    public TrackingEventDO toDO(TrackingEvent entity) {
        if (entity == null) {
            return null;
        }
        TrackingEventDO doObj = new TrackingEventDO();
        doObj.setId(entity.getId());
        doObj.setTrackingNo(entity.getTrackingNo());
        doObj.setEventType(entity.getEventType());
        doObj.setLocation(entity.getLocation());
        doObj.setDescription(entity.getDescription());
        doObj.setOperator(entity.getOperator());
        doObj.setOccurredAt(entity.getOccurredAt());
        doObj.setCreatedAt(entity.getCreatedAt());
        return doObj;
    }
}