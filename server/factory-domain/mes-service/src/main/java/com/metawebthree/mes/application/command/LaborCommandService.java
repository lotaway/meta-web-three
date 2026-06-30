package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.labor.*;
import com.metawebthree.mes.domain.repository.labor.*;
import com.metawebthree.mes.domain.service.labor.LaborDomainService;
import org.springframework.stereotype.Service;

import java.time.LocalDate;

@Service
public class LaborCommandService {

    private final OperatorRepository operatorRepository;
    private final WorkCenterAssignmentRepository assignmentRepository;
    private final TimeRecordRepository timeRecordRepository;
    private final AttendanceRepository attendanceRepository;
    private final LaborDomainService laborDomainService;

    public LaborCommandService(OperatorRepository operatorRepository,
                               WorkCenterAssignmentRepository assignmentRepository,
                               TimeRecordRepository timeRecordRepository,
                               AttendanceRepository attendanceRepository,
                               LaborDomainService laborDomainService) {
        this.operatorRepository = operatorRepository;
        this.assignmentRepository = assignmentRepository;
        this.timeRecordRepository = timeRecordRepository;
        this.attendanceRepository = attendanceRepository;
        this.laborDomainService = laborDomainService;
    }

    // ---- Operator ----
    public Operator createOperator(String operatorCode, String operatorName, String department, String shiftGroup) {
        Operator entity = new Operator();
        entity.create(operatorCode, operatorName, department, shiftGroup);
        return operatorRepository.save(entity);
    }

    public Operator updateOperator(Long id, String operatorName, String department, String jobTitle,
                                    String shiftGroup, String phone, String email) {
        Operator op = operatorRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Operator not found: " + id));
        op.setOperatorName(operatorName);
        op.setDepartment(department);
        op.setJobTitle(jobTitle);
        op.setShiftGroup(shiftGroup);
        op.setPhone(phone);
        op.setEmail(email);
        operatorRepository.update(op);
        return op;
    }

    public Operator changeOperatorStatus(Long id, String status) {
        Operator op = operatorRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Operator not found: " + id));
        Operator.OperatorStatus newStatus = Operator.OperatorStatus.valueOf(status);
        switch (newStatus) {
            case ACTIVE -> op.activate();
            case INACTIVE -> op.inactivate();
            case ON_LEAVE -> op.setOnLeave();
            case TERMINATED -> op.terminate();
        }
        operatorRepository.update(op);
        return op;
    }

    public Operator addSkill(Long operatorId, String skillCode, String skillName, String skillLevel) {
        Operator op = operatorRepository.findById(operatorId)
            .orElseThrow(() -> new IllegalArgumentException("Operator not found: " + operatorId));
        OperatorSkill skill = new OperatorSkill();
        skill.create(operatorId, skillCode, skillName, OperatorSkill.SkillLevel.valueOf(skillLevel));
        op.addSkill(skill);
        operatorRepository.update(op);
        return op;
    }

    public void deleteOperator(Long id) {
        operatorRepository.deleteById(id);
    }

    // ---- Attendance ----
    public Attendance clockIn(Long operatorId) {
        return laborDomainService.clockIn(operatorId);
    }

    public Attendance clockOut(Long operatorId) {
        return laborDomainService.clockOut(operatorId);
    }

    public Attendance markAttendance(Long operatorId, LocalDate date, String status) {
        Attendance attendance = attendanceRepository.findByOperatorIdAndDate(operatorId, date)
            .orElseThrow(() -> new IllegalArgumentException("Attendance not found"));
        switch (Attendance.AttendanceStatus.valueOf(status)) {
            case ABSENT -> attendance.markAbsent();
            case VACATION -> attendance.markVacation();
            case SICK -> attendance.markSick();
            default -> throw new IllegalArgumentException("Invalid manual status: " + status);
        }
        attendanceRepository.update(attendance);
        return attendance;
    }

    // ---- Time Record ----
    public TimeRecord startTimeRecord(Long operatorId, String recordType) {
        return laborDomainService.startTimeRecord(operatorId, recordType);
    }

    public TimeRecord endTimeRecord(Long recordId) {
        return laborDomainService.endTimeRecord(recordId);
    }

    public TimeRecord submitTimeRecord(Long recordId) {
        return laborDomainService.submitTimeRecord(recordId);
    }

    public TimeRecord approveTimeRecord(Long recordId, String approvedBy) {
        return laborDomainService.approveTimeRecord(recordId, approvedBy);
    }

    public TimeRecord rejectTimeRecord(Long recordId, String approvedBy) {
        return laborDomainService.rejectTimeRecord(recordId, approvedBy);
    }

    public TimeRecord assignTimeRecordToTask(Long recordId, String workOrderNo, String taskNo,
                                              String operationCode, String workCenterId) {
        TimeRecord record = timeRecordRepository.findById(recordId)
            .orElseThrow(() -> new IllegalArgumentException("Time record not found: " + recordId));
        record.assignToTask(workOrderNo, taskNo, operationCode, workCenterId);
        timeRecordRepository.update(record);
        return record;
    }

    public void deleteTimeRecord(Long id) {
        timeRecordRepository.deleteById(id);
    }

    // ---- Assignment ----
    public WorkCenterAssignment assignToWorkCenter(Long operatorId, String workCenterId,
                                                    String workCenterName, String shiftType) {
        return laborDomainService.assignToWorkCenter(operatorId, workCenterId, workCenterName, shiftType);
    }

    public void endAssignment(Long assignmentId) {
        laborDomainService.endAssignment(assignmentId);
    }
}
