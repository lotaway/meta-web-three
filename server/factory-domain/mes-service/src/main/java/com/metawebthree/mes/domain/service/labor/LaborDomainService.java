package com.metawebthree.mes.domain.service.labor;

import com.metawebthree.mes.domain.entity.labor.*;

public interface LaborDomainService {
    Attendance clockIn(Long operatorId);
    Attendance clockOut(Long operatorId);
    TimeRecord startTimeRecord(Long operatorId, String recordType);
    TimeRecord endTimeRecord(Long recordId);
    TimeRecord submitTimeRecord(Long recordId);
    TimeRecord approveTimeRecord(Long recordId, String approvedBy);
    TimeRecord rejectTimeRecord(Long recordId, String approvedBy);
    WorkCenterAssignment assignToWorkCenter(Long operatorId, String workCenterId,
                                             String workCenterName, String shiftType);
    void endAssignment(Long assignmentId);
}
