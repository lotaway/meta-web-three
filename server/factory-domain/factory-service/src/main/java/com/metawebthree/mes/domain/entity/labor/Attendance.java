package com.metawebthree.mes.domain.entity.labor;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

public class Attendance {

    public enum AttendanceStatus {
        PRESENT, LATE, ABSENT, HALF_DAY, OVERTIME, VACATION, SICK, BUSINESS_TRIP
    }

    private Long id;
    private Long operatorId;
    private String operatorCode;
    private String operatorName;
    private LocalDate attendanceDate;
    private LocalTime clockIn;
    private LocalTime clockOut;
    private LocalTime scheduledStart;
    private LocalTime scheduledEnd;
    private AttendanceStatus status;
    private boolean overtime;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(Long operatorId, String operatorCode, String operatorName,
                       LocalDate attendanceDate, LocalTime scheduledStart, LocalTime scheduledEnd) {
        this.operatorId = operatorId;
        this.operatorCode = operatorCode;
        this.operatorName = operatorName;
        this.attendanceDate = attendanceDate;
        this.scheduledStart = scheduledStart;
        this.scheduledEnd = scheduledEnd;
        this.status = AttendanceStatus.ABSENT;
        this.overtime = false;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void clockIn(LocalTime time) {
        this.clockIn = time;
        if (time.isAfter(scheduledStart.plusMinutes(15))) {
            this.status = AttendanceStatus.LATE;
        } else {
            this.status = AttendanceStatus.PRESENT;
        }
        this.updatedAt = LocalDateTime.now();
    }

    public void clockOut(LocalTime time) {
        this.clockOut = time;
        if (time.isAfter(scheduledEnd.plusHours(1))) {
            this.overtime = true;
        }
        if (clockIn != null) {
            this.status = AttendanceStatus.PRESENT;
        }
        this.updatedAt = LocalDateTime.now();
    }

    public void markAbsent() {
        this.status = AttendanceStatus.ABSENT;
        this.updatedAt = LocalDateTime.now();
    }

    public void markVacation() {
        this.status = AttendanceStatus.VACATION;
        this.updatedAt = LocalDateTime.now();
    }

    public void markSick() {
        this.status = AttendanceStatus.SICK;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getOperatorId() { return operatorId; }
    public void setOperatorId(Long operatorId) { this.operatorId = operatorId; }
    public String getOperatorCode() { return operatorCode; }
    public void setOperatorCode(String operatorCode) { this.operatorCode = operatorCode; }
    public String getOperatorName() { return operatorName; }
    public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
    public LocalDate getAttendanceDate() { return attendanceDate; }
    public void setAttendanceDate(LocalDate attendanceDate) { this.attendanceDate = attendanceDate; }
    public LocalTime getClockIn() { return clockIn; }
    public void setClockIn(LocalTime clockIn) { this.clockIn = clockIn; }
    public LocalTime getClockOut() { return clockOut; }
    public void setClockOut(LocalTime clockOut) { this.clockOut = clockOut; }
    public LocalTime getScheduledStart() { return scheduledStart; }
    public void setScheduledStart(LocalTime scheduledStart) { this.scheduledStart = scheduledStart; }
    public LocalTime getScheduledEnd() { return scheduledEnd; }
    public void setScheduledEnd(LocalTime scheduledEnd) { this.scheduledEnd = scheduledEnd; }
    public AttendanceStatus getStatus() { return status; }
    public void setStatus(AttendanceStatus status) { this.status = status; }
    public boolean isOvertime() { return overtime; }
    public void setOvertime(boolean overtime) { this.overtime = overtime; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
