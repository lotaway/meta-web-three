package com.metawebthree.mes.domain.repository.labor;

import com.metawebthree.mes.domain.entity.labor.TimeRecord;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface TimeRecordRepository {
    Optional<TimeRecord> findById(Long id);
    List<TimeRecord> findByOperatorId(Long operatorId);
    List<TimeRecord> findByOperatorIdAndDate(Long operatorId, LocalDate date);
    List<TimeRecord> findByDateRange(LocalDate start, LocalDate end);
    List<TimeRecord> findByWorkOrderNo(String workOrderNo);
    List<TimeRecord> findByStatus(TimeRecord.RecordStatus status);
    List<TimeRecord> findAll();
    TimeRecord save(TimeRecord record);
    void update(TimeRecord record);
    void deleteById(Long id);
}
