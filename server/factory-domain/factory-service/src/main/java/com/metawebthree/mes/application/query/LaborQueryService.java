package com.metawebthree.mes.application.query;

import com.metawebthree.mes.domain.entity.labor.*;
import com.metawebthree.mes.domain.repository.labor.*;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Service
public class LaborQueryService {

    private final OperatorRepository operatorRepository;
    private final WorkCenterAssignmentRepository assignmentRepository;
    private final TimeRecordRepository timeRecordRepository;
    private final AttendanceRepository attendanceRepository;

    public LaborQueryService(OperatorRepository operatorRepository,
                             WorkCenterAssignmentRepository assignmentRepository,
                             TimeRecordRepository timeRecordRepository,
                             AttendanceRepository attendanceRepository) {
        this.operatorRepository = operatorRepository;
        this.assignmentRepository = assignmentRepository;
        this.timeRecordRepository = timeRecordRepository;
        this.attendanceRepository = attendanceRepository;
    }

    // ---- Operator ----
    public Optional<Operator> findOperatorById(Long id) { return operatorRepository.findById(id); }
    public Optional<Operator> findOperatorByCode(String code) { return operatorRepository.findByOperatorCode(code); }
    public List<Operator> findOperatorsByDepartment(String department) { return operatorRepository.findByDepartment(department); }
    public List<Operator> findOperatorsByStatus(String status) { return operatorRepository.findByStatus(Operator.OperatorStatus.valueOf(status)); }
    public List<Operator> findAllOperators() { return operatorRepository.findAll(); }

    // ---- Assignment ----
    public Optional<WorkCenterAssignment> findAssignmentById(Long id) { return assignmentRepository.findById(id); }
    public List<WorkCenterAssignment> findAssignmentsByOperator(Long operatorId) { return assignmentRepository.findByOperatorId(operatorId); }
    public List<WorkCenterAssignment> findActiveAssignmentsByOperator(Long operatorId) { return assignmentRepository.findActiveByOperatorId(operatorId); }
    public List<WorkCenterAssignment> findAssignmentsByWorkCenter(String workCenterId) { return assignmentRepository.findByWorkCenterId(workCenterId); }
    public List<WorkCenterAssignment> findAllAssignments() { return assignmentRepository.findAll(); }

    // ---- Time Record ----
    public Optional<TimeRecord> findTimeRecordById(Long id) { return timeRecordRepository.findById(id); }
    public List<TimeRecord> findTimeRecordsByOperator(Long operatorId) { return timeRecordRepository.findByOperatorId(operatorId); }
    public List<TimeRecord> findTimeRecordsByDateRange(LocalDate start, LocalDate end) { return timeRecordRepository.findByDateRange(start, end); }
    public List<TimeRecord> findTimeRecordsByWorkOrder(String workOrderNo) { return timeRecordRepository.findByWorkOrderNo(workOrderNo); }
    public List<TimeRecord> findTimeRecordsByStatus(String status) { return timeRecordRepository.findByStatus(TimeRecord.RecordStatus.valueOf(status)); }
    public List<TimeRecord> findAllTimeRecords() { return timeRecordRepository.findAll(); }

    // ---- Attendance ----
    public Optional<Attendance> findAttendanceById(Long id) { return attendanceRepository.findById(id); }
    public List<Attendance> findAttendanceByOperator(Long operatorId) { return attendanceRepository.findByOperatorId(operatorId); }
    public List<Attendance> findAttendanceByDate(LocalDate date) { return attendanceRepository.findByDate(date); }
    public List<Attendance> findAttendanceByDateRange(LocalDate start, LocalDate end) { return attendanceRepository.findByDateRange(start, end); }
    public List<Attendance> findAllAttendance() { return attendanceRepository.findAll(); }
}
