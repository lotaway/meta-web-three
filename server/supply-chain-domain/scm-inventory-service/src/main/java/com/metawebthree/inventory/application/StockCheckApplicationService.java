package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.stockcheck.StockCheckPlanDTO;
import com.metawebthree.inventory.application.dto.stockcheck.StockCheckRecordDTO;
import com.metawebthree.inventory.application.dto.stockcheck.StockCheckDiffDTO;
import com.metawebthree.inventory.application.dto.stockcheck.StockCheckReportDTO;

import java.util.List;

public interface StockCheckApplicationService {

    // Plan management
    StockCheckPlanDTO createPlan(StockCheckPlanDTO dto);

    StockCheckPlanDTO updatePlan(Long id, StockCheckPlanDTO dto);

    void deletePlan(Long id);

    StockCheckPlanDTO approvePlan(Long id);

    StockCheckPlanDTO startPlan(Long id);

    StockCheckPlanDTO completePlan(Long id);

    StockCheckPlanDTO cancelPlan(Long id);

    StockCheckPlanDTO queryPlan(Long id);

    StockCheckPlanDTO queryPlanByNo(String planNo);

    List<StockCheckPlanDTO> listPlans(Long warehouseId, String status);

    // Record management
    StockCheckRecordDTO createRecord(StockCheckRecordDTO dto);

    StockCheckRecordDTO batchCreateRecords(List<StockCheckRecordDTO> records);

    StockCheckRecordDTO updateRecord(Long id, StockCheckRecordDTO dto);

    List<StockCheckRecordDTO> listRecords(Long planId, String status, Boolean hasDifference);

    // Diff management
    StockCheckDiffDTO approveDiff(Long id, String approver, String remark);

    StockCheckDiffDTO rejectDiff(Long id, String approver, String remark);

    StockCheckDiffDTO processDiff(Long id, String processor, String solution, String remark);

    List<StockCheckDiffDTO> listDiffs(Long planId, String approvalStatus, String processingStatus);

    List<StockCheckDiffDTO> listPendingApprovalDiffs();

    // Report
    StockCheckReportDTO generateReport(Long planId);

    List<StockCheckReportDTO> listReports(Long warehouseId, String startDate, String endDate);
}