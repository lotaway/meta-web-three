package com.metawebthree.mes.domain.repository.labor;

import com.metawebthree.mes.domain.entity.labor.Attendance;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface AttendanceRepository {
    Optional<Attendance> findById(Long id);
    Optional<Attendance> findByOperatorIdAndDate(Long operatorId, LocalDate date);
    List<Attendance> findByOperatorId(Long operatorId);
    List<Attendance> findByDate(LocalDate date);
    List<Attendance> findByDateRange(LocalDate start, LocalDate end);
    List<Attendance> findByStatus(Attendance.AttendanceStatus status);
    List<Attendance> findAll();
    Attendance save(Attendance attendance);
    void update(Attendance attendance);
    void deleteById(Long id);
}
