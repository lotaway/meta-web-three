package com.metawebthree.mes.domain.repository.scada;

import com.metawebthree.mes.domain.entity.scada.TelemetryRecord;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface TelemetryRecordRepository {
    Optional<TelemetryRecord> findById(Long id);
    List<TelemetryRecord> findByEquipmentCode(String equipmentCode);
    List<TelemetryRecord> findByEquipmentCodeAndTimeRange(String equipmentCode, LocalDateTime start, LocalDateTime end);
    List<TelemetryRecord> findByTopic(String topic);
    List<TelemetryRecord> findAllByTimeRange(LocalDateTime start, LocalDateTime end);
    TelemetryRecord save(TelemetryRecord record);
    void deleteById(Long id);
}
