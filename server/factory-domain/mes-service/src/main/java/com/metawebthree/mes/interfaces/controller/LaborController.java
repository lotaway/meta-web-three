package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.mes.application.command.LaborCommandService;
import com.metawebthree.mes.application.query.LaborQueryService;
import com.metawebthree.mes.domain.entity.labor.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;

@RestController
@RequestMapping("/api/mes/labor")
public class LaborController {

    private final LaborCommandService commandService;
    private final LaborQueryService queryService;

    public LaborController(LaborCommandService commandService, LaborQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    // ==================== Operator ====================

    @PostMapping("/operators")
    public ResponseEntity<Operator> createOperator(@RequestBody CreateOperatorRequest req) {
        return ResponseEntity.ok(commandService.createOperator(
            req.getOperatorCode(), req.getOperatorName(), req.getDepartment(), req.getShiftGroup()));
    }

    @PutMapping("/operators/{id}")
    public ResponseEntity<Operator> updateOperator(@PathVariable Long id, @RequestBody UpdateOperatorRequest req) {
        return ResponseEntity.ok(commandService.updateOperator(
            id, req.getOperatorName(), req.getDepartment(), req.getJobTitle(),
            req.getShiftGroup(), req.getPhone(), req.getEmail()));
    }

    @PutMapping("/operators/{id}/status")
    public ResponseEntity<Operator> changeStatus(@PathVariable Long id, @RequestParam String status) {
        return ResponseEntity.ok(commandService.changeOperatorStatus(id, status));
    }

    @PostMapping("/operators/{id}/skills")
    public ResponseEntity<Operator> addSkill(@PathVariable Long id, @RequestBody AddSkillRequest req) {
        return ResponseEntity.ok(commandService.addSkill(id, req.getSkillCode(), req.getSkillName(), req.getSkillLevel()));
    }

    @DeleteMapping("/operators/{id}")
    public ResponseEntity<Void> deleteOperator(@PathVariable Long id) {
        commandService.deleteOperator(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/operators/{id}")
    public ResponseEntity<Operator> getOperator(@PathVariable Long id) {
        return queryService.findOperatorById(id)
            .map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/operators")
    public ResponseEntity<List<Operator>> listOperators(
            @RequestParam(required = false) String department,
            @RequestParam(required = false) String status) {
        if (department != null) return ResponseEntity.ok(queryService.findOperatorsByDepartment(department));
        if (status != null) return ResponseEntity.ok(queryService.findOperatorsByStatus(status));
        return ResponseEntity.ok(queryService.findAllOperators());
    }

    // ==================== Attendance ====================

    @PostMapping("/attendance/clock-in")
    public ResponseEntity<Attendance> clockIn(@RequestParam Long operatorId) {
        return ResponseEntity.ok(commandService.clockIn(operatorId));
    }

    @PostMapping("/attendance/clock-out")
    public ResponseEntity<Attendance> clockOut(@RequestParam Long operatorId) {
        return ResponseEntity.ok(commandService.clockOut(operatorId));
    }

    @PutMapping("/attendance/{id}/status")
    public ResponseEntity<Attendance> markAttendance(@PathVariable Long id,
                                                      @RequestParam LocalDate date,
                                                      @RequestParam String status) {
        return ResponseEntity.ok(commandService.markAttendance(id, date, status));
    }

    @GetMapping("/attendance")
    public ResponseEntity<List<Attendance>> listAttendance(
            @RequestParam(required = false) Long operatorId,
            @RequestParam(required = false) String date) {
        if (operatorId != null) return ResponseEntity.ok(queryService.findAttendanceByOperator(operatorId));
        if (date != null) return ResponseEntity.ok(queryService.findAttendanceByDate(LocalDate.parse(date)));
        return ResponseEntity.ok(queryService.findAllAttendance());
    }

    // ==================== Time Record ====================

    @PostMapping("/time-records/start")
    public ResponseEntity<TimeRecord> startTimeRecord(@RequestParam Long operatorId,
                                                       @RequestParam(defaultValue = "REGULAR") String recordType) {
        return ResponseEntity.ok(commandService.startTimeRecord(operatorId, recordType));
    }

    @PostMapping("/time-records/{id}/end")
    public ResponseEntity<TimeRecord> endTimeRecord(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.endTimeRecord(id));
    }

    @PostMapping("/time-records/{id}/submit")
    public ResponseEntity<TimeRecord> submitTimeRecord(@PathVariable Long id) {
        return ResponseEntity.ok(commandService.submitTimeRecord(id));
    }

    @PostMapping("/time-records/{id}/approve")
    public ResponseEntity<TimeRecord> approveTimeRecord(@PathVariable Long id,
                                                         @RequestParam String approvedBy) {
        return ResponseEntity.ok(commandService.approveTimeRecord(id, approvedBy));
    }

    @PostMapping("/time-records/{id}/reject")
    public ResponseEntity<TimeRecord> rejectTimeRecord(@PathVariable Long id,
                                                        @RequestParam String approvedBy) {
        return ResponseEntity.ok(commandService.rejectTimeRecord(id, approvedBy));
    }

    @PutMapping("/time-records/{id}/assign-task")
    public ResponseEntity<TimeRecord> assignToTask(@PathVariable Long id,
                                                    @RequestBody AssignTaskRequest req) {
        return ResponseEntity.ok(commandService.assignTimeRecordToTask(
            id, req.getWorkOrderNo(), req.getTaskNo(), req.getOperationCode(), req.getWorkCenterId()));
    }

    @DeleteMapping("/time-records/{id}")
    public ResponseEntity<Void> deleteTimeRecord(@PathVariable Long id) {
        commandService.deleteTimeRecord(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/time-records/{id}")
    public ResponseEntity<TimeRecord> getTimeRecord(@PathVariable Long id) {
        return queryService.findTimeRecordById(id)
            .map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/time-records")
    public ResponseEntity<List<TimeRecord>> listTimeRecords(
            @RequestParam(required = false) Long operatorId,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate) {
        if (operatorId != null) return ResponseEntity.ok(queryService.findTimeRecordsByOperator(operatorId));
        if (status != null) return ResponseEntity.ok(queryService.findTimeRecordsByStatus(status));
        if (startDate != null && endDate != null)
            return ResponseEntity.ok(queryService.findTimeRecordsByDateRange(LocalDate.parse(startDate), LocalDate.parse(endDate)));
        return ResponseEntity.ok(queryService.findAllTimeRecords());
    }

    // ==================== Assignment ====================

    @PostMapping("/assignments")
    public ResponseEntity<WorkCenterAssignment> createAssignment(@RequestBody CreateAssignmentRequest req) {
        return ResponseEntity.ok(commandService.assignToWorkCenter(
            req.getOperatorId(), req.getWorkCenterId(), req.getWorkCenterName(), req.getShiftType()));
    }

    @PostMapping("/assignments/{id}/end")
    public ResponseEntity<Void> endAssignment(@PathVariable Long id) {
        commandService.endAssignment(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/assignments")
    public ResponseEntity<List<WorkCenterAssignment>> listAssignments(
            @RequestParam(required = false) Long operatorId,
            @RequestParam(required = false) String workCenterId) {
        if (operatorId != null) return ResponseEntity.ok(queryService.findActiveAssignmentsByOperator(operatorId));
        if (workCenterId != null) return ResponseEntity.ok(queryService.findAssignmentsByWorkCenter(workCenterId));
        return ResponseEntity.ok(queryService.findAllAssignments());
    }

    // ==================== Inner DTOs ====================

    public static class CreateOperatorRequest {
        private String operatorCode;
        private String operatorName;
        private String department;
        private String shiftGroup;
        public String getOperatorCode() { return operatorCode; }
        public void setOperatorCode(String operatorCode) { this.operatorCode = operatorCode; }
        public String getOperatorName() { return operatorName; }
        public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
        public String getDepartment() { return department; }
        public void setDepartment(String department) { this.department = department; }
        public String getShiftGroup() { return shiftGroup; }
        public void setShiftGroup(String shiftGroup) { this.shiftGroup = shiftGroup; }
    }

    public static class UpdateOperatorRequest {
        private String operatorName;
        private String department;
        private String jobTitle;
        private String shiftGroup;
        private String phone;
        private String email;
        public String getOperatorName() { return operatorName; }
        public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
        public String getDepartment() { return department; }
        public void setDepartment(String department) { this.department = department; }
        public String getJobTitle() { return jobTitle; }
        public void setJobTitle(String jobTitle) { this.jobTitle = jobTitle; }
        public String getShiftGroup() { return shiftGroup; }
        public void setShiftGroup(String shiftGroup) { this.shiftGroup = shiftGroup; }
        public String getPhone() { return phone; }
        public void setPhone(String phone) { this.phone = phone; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
    }

    public static class AddSkillRequest {
        private String skillCode;
        private String skillName;
        private String skillLevel;
        public String getSkillCode() { return skillCode; }
        public void setSkillCode(String skillCode) { this.skillCode = skillCode; }
        public String getSkillName() { return skillName; }
        public void setSkillName(String skillName) { this.skillName = skillName; }
        public String getSkillLevel() { return skillLevel; }
        public void setSkillLevel(String skillLevel) { this.skillLevel = skillLevel; }
    }

    public static class AssignTaskRequest {
        private String workOrderNo;
        private String taskNo;
        private String operationCode;
        private String workCenterId;
        public String getWorkOrderNo() { return workOrderNo; }
        public void setWorkOrderNo(String workOrderNo) { this.workOrderNo = workOrderNo; }
        public String getTaskNo() { return taskNo; }
        public void setTaskNo(String taskNo) { this.taskNo = taskNo; }
        public String getOperationCode() { return operationCode; }
        public void setOperationCode(String operationCode) { this.operationCode = operationCode; }
        public String getWorkCenterId() { return workCenterId; }
        public void setWorkCenterId(String workCenterId) { this.workCenterId = workCenterId; }
    }

    public static class CreateAssignmentRequest {
        private Long operatorId;
        private String workCenterId;
        private String workCenterName;
        private String shiftType;
        public Long getOperatorId() { return operatorId; }
        public void setOperatorId(Long operatorId) { this.operatorId = operatorId; }
        public String getWorkCenterId() { return workCenterId; }
        public void setWorkCenterId(String workCenterId) { this.workCenterId = workCenterId; }
        public String getWorkCenterName() { return workCenterName; }
        public void setWorkCenterName(String workCenterName) { this.workCenterName = workCenterName; }
        public String getShiftType() { return shiftType; }
        public void setShiftType(String shiftType) { this.shiftType = shiftType; }
    }
}
