package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.stockcheck.*;
import com.metawebthree.inventory.domain.entity.stockcheck.*;
import com.metawebthree.inventory.domain.repository.stockcheck.*;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class StockCheckApplicationServiceImpl implements StockCheckApplicationService {

    private static final String PLAN_NO_PREFIX = "CK";

    private final StockCheckPlanRepository planRepository;
    private final StockCheckRecordRepository recordRepository;
    private final StockCheckDiffRepository diffRepository;
    private final InventoryRepository inventoryRepository;

    @Override
    @Transactional
    public StockCheckPlanDTO createPlan(StockCheckPlanDTO dto) {
        StockCheckPlan plan = new StockCheckPlan();
        plan.setPlanNo(generatePlanNo());
        plan.setPlanName(dto.getPlanName());
        plan.setCheckType(dto.getCheckType() != null ? dto.getCheckType() : StockCheckPlan.TYPE_FULL);
        plan.setWarehouseId(dto.getWarehouseId());
        plan.setWarehouseName(dto.getWarehouseName());
        plan.setStatus(StockCheckPlan.STATUS_DRAFT);
        plan.setPlannedStartTime(dto.getPlannedStartTime());
        plan.setPlannedEndTime(dto.getPlannedEndTime());
        plan.setCreator(dto.getCreator());
        plan.setCreateTime(LocalDateTime.now());
        plan.setRemark(dto.getRemark());
        plan.setDeleted(false);

        planRepository.save(plan);
        log.info("Stock check plan created: planNo={}, warehouseId={}", plan.getPlanNo(), plan.getWarehouseId());

        return toPlanDTO(plan);
    }

    @Override
    @Transactional
    public StockCheckPlanDTO updatePlan(Long id, StockCheckPlanDTO dto) {
        StockCheckPlan plan = planRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + id));

        if (!plan.isEditable()) {
            throw new IllegalStateException("Only draft plan can be updated");
        }

        if (dto.getPlanName() != null) {
            plan.setPlanName(dto.getPlanName());
        }
        if (dto.getCheckType() != null) {
            plan.setCheckType(dto.getCheckType());
        }
        if (dto.getWarehouseId() != null) {
            plan.setWarehouseId(dto.getWarehouseId());
        }
        if (dto.getPlannedStartTime() != null) {
            plan.setPlannedStartTime(dto.getPlannedStartTime());
        }
        if (dto.getPlannedEndTime() != null) {
            plan.setPlannedEndTime(dto.getPlannedEndTime());
        }
        if (dto.getRemark() != null) {
            plan.setRemark(dto.getRemark());
        }
        plan.setUpdater(dto.getCreator());
        plan.setUpdateTime(LocalDateTime.now());

        planRepository.save(plan);
        log.info("Stock check plan updated: planNo={}", plan.getPlanNo());

        return toPlanDTO(plan);
    }

    @Override
    @Transactional
    public void deletePlan(Long id) {
        StockCheckPlan plan = planRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + id));

        if (!plan.isEditable()) {
            throw new IllegalStateException("Only draft plan can be deleted");
        }

        plan.setDeleted(true);
        planRepository.save(plan);
        log.info("Stock check plan deleted: planNo={}", plan.getPlanNo());
    }

    @Override
    @Transactional
    public StockCheckPlanDTO approvePlan(Long id) {
        StockCheckPlan plan = planRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + id));

        plan.approve();
        plan.setUpdater(plan.getCreator());
        plan.setUpdateTime(LocalDateTime.now());

        planRepository.save(plan);
        log.info("Stock check plan approved: planNo={}", plan.getPlanNo());

        return toPlanDTO(plan);
    }

    @Override
    @Transactional
    public StockCheckPlanDTO startPlan(Long id) {
        StockCheckPlan plan = planRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + id));

        plan.start();
        plan.setUpdater(plan.getCreator());
        plan.setUpdateTime(LocalDateTime.now());

        planRepository.save(plan);
        log.info("Stock check plan started: planNo={}", plan.getPlanNo());

        return toPlanDTO(plan);
    }

    @Override
    @Transactional
    public StockCheckPlanDTO completePlan(Long id) {
        StockCheckPlan plan = planRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + id));

        plan.complete();
        plan.setUpdater(plan.getCreator());
        plan.setUpdateTime(LocalDateTime.now());

        planRepository.save(plan);
        log.info("Stock check plan completed: planNo={}", plan.getPlanNo());

        return toPlanDTO(plan);
    }

    @Override
    @Transactional
    public StockCheckPlanDTO cancelPlan(Long id) {
        StockCheckPlan plan = planRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + id));

        plan.cancel();
        plan.setUpdater(plan.getCreator());
        plan.setUpdateTime(LocalDateTime.now());

        planRepository.save(plan);
        log.info("Stock check plan cancelled: planNo={}", plan.getPlanNo());

        return toPlanDTO(plan);
    }

    @Override
    public StockCheckPlanDTO queryPlan(Long id) {
        return planRepository.findById(id)
                .map(this::toPlanDTO)
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + id));
    }

    @Override
    public StockCheckPlanDTO queryPlanByNo(String planNo) {
        return planRepository.findByPlanNo(planNo)
                .map(this::toPlanDTO)
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + planNo));
    }

    @Override
    public List<StockCheckPlanDTO> listPlans(Long warehouseId, String status) {
        List<StockCheckPlan> plans;
        if (warehouseId != null && status != null) {
            plans = planRepository.findByWarehouseIdAndStatus(warehouseId, status);
        } else if (warehouseId != null) {
            plans = planRepository.findByWarehouseId(warehouseId);
        } else if (status != null) {
            plans = planRepository.findByStatus(status);
        } else {
            plans = planRepository.findAll();
        }

        return plans.stream()
                .map(this::toPlanDTO)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional
    public StockCheckRecordDTO createRecord(StockCheckRecordDTO dto) {
        StockCheckPlan plan = planRepository.findById(dto.getPlanId())
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + dto.getPlanId()));

        if (!StockCheckPlan.STATUS_IN_PROGRESS.equals(plan.getStatus())) {
            throw new IllegalStateException("Only in-progress plan can accept records");
        }

        StockCheckRecord record = new StockCheckRecord();
        record.setPlanId(dto.getPlanId());
        record.setPlanNo(plan.getPlanNo());
        record.setSkuCode(dto.getSkuCode());
        record.setProductName(dto.getProductName());
        record.setLocationCode(dto.getLocationCode());
        record.setWarehouseId(dto.getWarehouseId());
        record.setBookQuantity(dto.getBookQuantity());
        record.setCheckQuantity(dto.getCheckQuantity());
        record.setStatus(StockCheckRecord.STATUS_PENDING);
        record.setChecker(dto.getChecker());
        record.setCheckTime(dto.getCheckTime() != null ? dto.getCheckTime() : LocalDateTime.now());
        record.setRemark(dto.getRemark());
        record.setCreator(dto.getChecker());
        record.setCreateTime(LocalDateTime.now());
        record.setDeleted(false);

        record.calculateDifference();
        recordRepository.save(record);

        log.info("Stock check record created: planNo={}, skuCode={}, diff={}", 
                record.getPlanNo(), record.getSkuCode(), record.getDifferenceQuantity());

        // Create diff record if there's a difference
        if (record.hasDifference()) {
            createDiffFromRecord(record);
        }

        return toRecordDTO(record);
    }

    @Override
    @Transactional
    public StockCheckRecordDTO batchCreateRecords(List<StockCheckRecordDTO> records) {
        // Process first record for now, batch processing would need more complex logic
        if (records != null && !records.isEmpty()) {
            return createRecord(records.get(0));
        }
        return null;
    }

    @Override
    @Transactional
    public StockCheckRecordDTO updateRecord(Long id, StockCheckRecordDTO dto) {
        StockCheckRecord record = recordRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Record not found: " + id));

        if (!StockCheckRecord.STATUS_PENDING.equals(record.getStatus())) {
            throw new IllegalStateException("Only pending record can be updated");
        }

        if (dto.getCheckQuantity() != null) {
            record.setCheckQuantity(dto.getCheckQuantity());
            record.calculateDifference();
        }
        if (dto.getRemark() != null) {
            record.setRemark(dto.getRemark());
        }
        record.setUpdater(dto.getChecker());
        record.setUpdateTime(LocalDateTime.now());

        recordRepository.save(record);
        log.info("Stock check record updated: id={}, skuCode={}", id, record.getSkuCode());

        return toRecordDTO(record);
    }

    @Override
    public List<StockCheckRecordDTO> listRecords(Long planId, String status, Boolean hasDifference) {
        List<StockCheckRecord> records;

        if (planId != null) {
            if (Boolean.TRUE.equals(hasDifference)) {
                records = recordRepository.findHasDifference(planId);
            } else if (status != null) {
                records = recordRepository.findByPlanId(planId).stream()
                        .filter(r -> status.equals(r.getStatus()))
                        .collect(Collectors.toList());
            } else {
                records = recordRepository.findByPlanId(planId);
            }
        } else {
            records = List.of();
        }

        return records.stream()
                .map(this::toRecordDTO)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional
    public StockCheckDiffDTO approveDiff(Long id, String approver, String remark) {
        StockCheckDiff diff = diffRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Diff not found: " + id));

        diff.approve(approver, remark);
        diff.setUpdater(approver);
        diff.setUpdateTime(LocalDateTime.now());

        diffRepository.save(diff);
        log.info("Stock check diff approved: id={}, approver={}", id, approver);

        return toDiffDTO(diff);
    }

    @Override
    @Transactional
    public StockCheckDiffDTO rejectDiff(Long id, String approver, String remark) {
        StockCheckDiff diff = diffRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Diff not found: " + id));

        diff.reject(approver, remark);
        diff.setUpdater(approver);
        diff.setUpdateTime(LocalDateTime.now());

        diffRepository.save(diff);
        log.info("Stock check diff rejected: id={}, approver={}", id, approver);

        return toDiffDTO(diff);
    }

    @Override
    @Transactional
    public StockCheckDiffDTO processDiff(Long id, String processor, String solution, String remark) {
        StockCheckDiff diff = diffRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Diff not found: " + id));

        diff.process(processor, solution, remark);
        diff.setUpdater(processor);
        diff.setUpdateTime(LocalDateTime.now());

        diffRepository.save(diff);
        log.info("Stock check diff processed: id={}, processor={}, solution={}", id, processor, solution);

        return toDiffDTO(diff);
    }

    @Override
    public List<StockCheckDiffDTO> listDiffs(Long planId, String approvalStatus, String processingStatus) {
        List<StockCheckDiff> diffs;

        if (planId != null) {
            diffs = diffRepository.findByPlanId(planId);
            if (approvalStatus != null) {
                diffs = diffs.stream()
                        .filter(d -> approvalStatus.equals(d.getApprovalStatus()))
                        .collect(Collectors.toList());
            }
            if (processingStatus != null) {
                diffs = diffs.stream()
                        .filter(d -> processingStatus.equals(d.getProcessingStatus()))
                        .collect(Collectors.toList());
            }
        } else {
            diffs = List.of();
        }

        return diffs.stream()
                .map(this::toDiffDTO)
                .collect(Collectors.toList());
    }

    @Override
    public List<StockCheckDiffDTO> listPendingApprovalDiffs() {
        return diffRepository.findPendingApproval().stream()
                .map(this::toDiffDTO)
                .collect(Collectors.toList());
    }

    @Override
    public StockCheckReportDTO generateReport(Long planId) {
        StockCheckPlan plan = planRepository.findById(planId)
                .orElseThrow(() -> new IllegalArgumentException("Plan not found: " + planId));

        List<StockCheckRecord> records = recordRepository.findByPlanId(planId);
        List<StockCheckDiff> diffs = diffRepository.findByPlanId(planId);

        StockCheckReportDTO report = new StockCheckReportDTO();
        report.setWarehouseId(plan.getWarehouseId());
        report.setWarehouseName(plan.getWarehouseName());
        report.setPlanNo(plan.getPlanNo());
        report.setPlanName(plan.getPlanName());
        report.setCheckType(plan.getCheckType());

        // Calculate statistics
        report.setTotalSkus(records.size());
        report.setCheckedSkus((int) records.stream()
                .filter(r -> r.getCheckQuantity() != null)
                .count());
        report.setDifferenceCount((int) records.stream()
                .filter(StockCheckRecord::hasDifference)
                .count());
        report.setShortCount((int) records.stream()
                .filter(r -> StockCheckRecord.DIFF_TYPE_SHORT.equals(r.getDifferenceType()))
                .count());
        report.setOverCount((int) records.stream()
                .filter(r -> StockCheckRecord.DIFF_TYPE_OVER.equals(r.getDifferenceType()))
                .count());

        // Calculate quantities
        report.setTotalBookQuantity(records.stream()
                .filter(r -> r.getBookQuantity() != null)
                .map(StockCheckRecord::getBookQuantity)
                .reduce(BigDecimal.ZERO, BigDecimal::add));
        report.setTotalCheckQuantity(records.stream()
                .filter(r -> r.getCheckQuantity() != null)
                .map(StockCheckRecord::getCheckQuantity)
                .reduce(BigDecimal.ZERO, BigDecimal::add));
        report.setTotalDifferenceQuantity(records.stream()
                .filter(r -> r.getDifferenceQuantity() != null)
                .map(StockCheckRecord::getDifferenceQuantity)
                .reduce(BigDecimal.ZERO, BigDecimal::add));

        // Approval statistics
        report.setPendingApprovalCount((int) diffs.stream()
                .filter(d -> StockCheckDiff.APPROVAL_STATUS_PENDING.equals(d.getApprovalStatus()))
                .count());
        report.setApprovedCount((int) diffs.stream()
                .filter(d -> StockCheckDiff.APPROVAL_STATUS_APPROVED.equals(d.getApprovalStatus()))
                .count());
        report.setRejectedCount((int) diffs.stream()
                .filter(d -> StockCheckDiff.APPROVAL_STATUS_REJECTED.equals(d.getApprovalStatus()))
                .count());
        report.setProcessedCount((int) diffs.stream()
                .filter(d -> StockCheckDiff.PROCESS_STATUS_PROCESSED.equals(d.getProcessingStatus()))
                .count());

        // Calculate rates
        if (report.getTotalSkus() > 0) {
            report.setCompletionRate(String.format("%.2f%%", 
                    (double) report.getCheckedSkus() / report.getTotalSkus() * 100));
            report.setAccuracyRate(String.format("%.2f%%", 
                    (double) (report.getTotalSkus() - report.getDifferenceCount()) / report.getTotalSkus() * 100));
        } else {
            report.setCompletionRate("0.00%");
            report.setAccuracyRate("0.00%");
        }

        return report;
    }

    @Override
    public List<StockCheckReportDTO> listReports(Long warehouseId, String startDate, String endDate) {
        // For now, return empty list - full implementation would query by date range
        return List.of();
    }

    private void createDiffFromRecord(StockCheckRecord record) {
        StockCheckDiff diff = new StockCheckDiff();
        diff.setRecordId(record.getId());
        diff.setPlanId(record.getPlanId());
        diff.setPlanNo(record.getPlanNo());
        diff.setSkuCode(record.getSkuCode());
        diff.setProductName(record.getProductName());
        diff.setLocationCode(record.getLocationCode());
        diff.setWarehouseId(record.getWarehouseId());
        diff.setBookQuantity(record.getBookQuantity());
        diff.setCheckQuantity(record.getCheckQuantity());
        diff.setDifferenceQuantity(record.getDifferenceQuantity());
        diff.setDifferenceType(record.getDifferenceType());
        diff.setProcessingStatus(StockCheckDiff.PROCESS_STATUS_PENDING);
        diff.setApprovalStatus(StockCheckDiff.APPROVAL_STATUS_PENDING);
        diff.setCreator(record.getCreator());
        diff.setCreateTime(LocalDateTime.now());
        diff.setDeleted(false);

        diffRepository.save(diff);
        log.info("Stock check diff created from record: recordId={}, diffType={}", 
                record.getId(), diff.getDifferenceType());
    }

    private String generatePlanNo() {
        return PLAN_NO_PREFIX + System.currentTimeMillis() 
                + UUID.randomUUID().toString().substring(0, 4).toUpperCase();
    }

    private StockCheckPlanDTO toPlanDTO(StockCheckPlan plan) {
        StockCheckPlanDTO dto = new StockCheckPlanDTO();
        dto.setId(plan.getId());
        dto.setPlanNo(plan.getPlanNo());
        dto.setPlanName(plan.getPlanName());
        dto.setCheckType(plan.getCheckType());
        dto.setWarehouseId(plan.getWarehouseId());
        dto.setWarehouseName(plan.getWarehouseName());
        dto.setStatus(plan.getStatus());
        dto.setPlannedStartTime(plan.getPlannedStartTime());
        dto.setPlannedEndTime(plan.getPlannedEndTime());
        dto.setActualStartTime(plan.getActualStartTime());
        dto.setActualEndTime(plan.getActualEndTime());
        dto.setCreator(plan.getCreator());
        dto.setCreateTime(plan.getCreateTime());
        dto.setRemark(plan.getRemark());
        return dto;
    }

    private StockCheckRecordDTO toRecordDTO(StockCheckRecord record) {
        StockCheckRecordDTO dto = new StockCheckRecordDTO();
        dto.setId(record.getId());
        dto.setPlanId(record.getPlanId());
        dto.setPlanNo(record.getPlanNo());
        dto.setSkuCode(record.getSkuCode());
        dto.setProductName(record.getProductName());
        dto.setLocationCode(record.getLocationCode());
        dto.setWarehouseId(record.getWarehouseId());
        dto.setBookQuantity(record.getBookQuantity());
        dto.setCheckQuantity(record.getCheckQuantity());
        dto.setDifferenceQuantity(record.getDifferenceQuantity());
        dto.setDifferenceType(record.getDifferenceType());
        dto.setStatus(record.getStatus());
        dto.setChecker(record.getChecker());
        dto.setCheckTime(record.getCheckTime());
        dto.setRemark(record.getRemark());
        return dto;
    }

    private StockCheckDiffDTO toDiffDTO(StockCheckDiff diff) {
        StockCheckDiffDTO dto = new StockCheckDiffDTO();
        dto.setId(diff.getId());
        dto.setRecordId(diff.getRecordId());
        dto.setPlanId(diff.getPlanId());
        dto.setPlanNo(diff.getPlanNo());
        dto.setSkuCode(diff.getSkuCode());
        dto.setProductName(diff.getProductName());
        dto.setLocationCode(diff.getLocationCode());
        dto.setWarehouseId(diff.getWarehouseId());
        dto.setBookQuantity(diff.getBookQuantity());
        dto.setCheckQuantity(diff.getCheckQuantity());
        dto.setDifferenceQuantity(diff.getDifferenceQuantity());
        dto.setDifferenceType(diff.getDifferenceType());
        dto.setProcessingStatus(diff.getProcessingStatus());
        dto.setApprovalStatus(diff.getApprovalStatus());
        dto.setApprover(diff.getApprover());
        dto.setApprovalTime(diff.getApprovalTime());
        dto.setApprovalRemark(diff.getApprovalRemark());
        dto.setSolution(diff.getSolution());
        dto.setProcessor(diff.getProcessor());
        dto.setProcessTime(diff.getProcessTime());
        dto.setProcessRemark(diff.getProcessRemark());
        dto.setNeedsApproval(diff.needsApproval());
        return dto;
    }
}