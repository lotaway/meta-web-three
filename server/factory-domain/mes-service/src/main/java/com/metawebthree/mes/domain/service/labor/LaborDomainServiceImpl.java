package com.metawebthree.mes.domain.service.labor;

import com.metawebthree.mes.domain.entity.labor.*;
import com.metawebthree.mes.domain.repository.labor.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

@Slf4j
@Service
public class LaborDomainServiceImpl implements LaborDomainService {

    private final OperatorRepository operatorRepository;
    private final AttendanceRepository attendanceRepository;
    private final TimeRecordRepository timeRecordRepository;
    private final WorkCenterAssignmentRepository assignmentRepository;

    public LaborDomainServiceImpl(OperatorRepository operatorRepository,
                                  AttendanceRepository attendanceRepository,
                                  TimeRecordRepository timeRecordRepository,
                                  WorkCenterAssignmentRepository assignmentRepository) {
        this.operatorRepository = operatorRepository;
        this.attendanceRepository = attendanceRepository;
        this.timeRecordRepository = timeRecordRepository;
        this.assignmentRepository = assignmentRepository;
    }

    @Override
    public Attendance clockIn(Long operatorId) {
        Operator operator = operatorRepository.findById(operatorId)
            .orElseThrow(() -> new IllegalArgumentException("Operator not found: " + operatorId));
        if (operator.getStatus() != Operator.OperatorStatus.ACTIVE) {
            throw new IllegalStateException("Operator is not active");
        }
        LocalDate today = LocalDate.now();
        Attendance attendance = attendanceRepository.findByOperatorIdAndDate(operatorId, today)
            .orElseGet(() -> {
                Attendance a = new Attendance();
                a.create(operatorId, operator.getOperatorCode(), operator.getOperatorName(),
                    today, LocalTime.of(8, 0), LocalTime.of(17, 0));
                return attendanceRepository.save(a);
            });
        if (attendance.getClockIn() != null) {
            throw new IllegalStateException("Already clocked in today");
        }
        attendance.clockIn(LocalTime.now());
        attendanceRepository.update(attendance);
        log.info("Operator {} clocked in at {}", operator.getOperatorCode(), attendance.getClockIn());
        return attendance;
    }

    @Override
    public Attendance clockOut(Long operatorId) {
        LocalDate today = LocalDate.now();
        Attendance attendance = attendanceRepository.findByOperatorIdAndDate(operatorId, today)
            .orElseThrow(() -> new IllegalStateException("No clock-in record found for today"));
        if (attendance.getClockOut() != null) {
            throw new IllegalStateException("Already clocked out today");
        }
        attendance.clockOut(LocalTime.now());
        attendanceRepository.update(attendance);
        log.info("Operator {} clocked out at {}", attendance.getOperatorCode(), attendance.getClockOut());
        return attendance;
    }

    @Override
    public TimeRecord startTimeRecord(Long operatorId, String recordType) {
        Operator operator = operatorRepository.findById(operatorId)
            .orElseThrow(() -> new IllegalArgumentException("Operator not found: " + operatorId));
        TimeRecord record = new TimeRecord();
        record.create(operatorId, operator.getOperatorCode(), operator.getOperatorName(),
            LocalDate.now(), LocalDateTime.now(), TimeRecord.RecordType.valueOf(recordType));
        return timeRecordRepository.save(record);
    }

    @Override
    public TimeRecord endTimeRecord(Long recordId) {
        TimeRecord record = timeRecordRepository.findById(recordId)
            .orElseThrow(() -> new IllegalArgumentException("Time record not found: " + recordId));
        record.clockOut(LocalDateTime.now());
        timeRecordRepository.update(record);
        return record;
    }

    @Override
    public TimeRecord submitTimeRecord(Long recordId) {
        TimeRecord record = timeRecordRepository.findById(recordId)
            .orElseThrow(() -> new IllegalArgumentException("Time record not found: " + recordId));
        record.submit();
        timeRecordRepository.update(record);
        return record;
    }

    @Override
    public TimeRecord approveTimeRecord(Long recordId, String approvedBy) {
        TimeRecord record = timeRecordRepository.findById(recordId)
            .orElseThrow(() -> new IllegalArgumentException("Time record not found: " + recordId));
        record.approve(approvedBy);
        timeRecordRepository.update(record);
        return record;
    }

    @Override
    public TimeRecord rejectTimeRecord(Long recordId, String approvedBy) {
        TimeRecord record = timeRecordRepository.findById(recordId)
            .orElseThrow(() -> new IllegalArgumentException("Time record not found: " + recordId));
        record.reject(approvedBy);
        timeRecordRepository.update(record);
        return record;
    }

    @Override
    public WorkCenterAssignment assignToWorkCenter(Long operatorId, String workCenterId,
                                                    String workCenterName, String shiftType) {
        WorkCenterAssignment assignment = new WorkCenterAssignment();
        assignment.create(operatorId, workCenterId, workCenterName, LocalDate.now(),
            WorkCenterAssignment.ShiftType.valueOf(shiftType));
        return assignmentRepository.save(assignment);
    }

    @Override
    public void endAssignment(Long assignmentId) {
        WorkCenterAssignment assignment = assignmentRepository.findById(assignmentId)
            .orElseThrow(() -> new IllegalArgumentException("Assignment not found: " + assignmentId));
        assignment.deactivate();
        assignmentRepository.update(assignment);
    }
}
