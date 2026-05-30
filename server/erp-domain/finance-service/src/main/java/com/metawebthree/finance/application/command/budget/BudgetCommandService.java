package com.metawebthree.finance.application.command.budget;

import com.metawebthree.finance.application.command.budget.dto.BudgetAdjustmentCreateCommand;
import com.metawebthree.finance.application.command.budget.dto.BudgetCreateCommand;
import com.metawebthree.finance.application.command.budget.dto.BudgetLineCreateCommand;
import com.metawebthree.finance.application.command.budget.dto.BudgetUpdateCommand;
import com.metawebthree.finance.domain.entity.budget.Budget;
import com.metawebthree.finance.domain.entity.budget.BudgetAdjustment;
import com.metawebthree.finance.domain.entity.budget.BudgetLine;
import com.metawebthree.finance.domain.repository.budget.BudgetRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.List;

@Service
public class BudgetCommandService {
    private final BudgetRepository budgetRepository;

    public BudgetCommandService(BudgetRepository budgetRepository) {
        this.budgetRepository = budgetRepository;
    }

    @Transactional
    public Long createBudget(BudgetCreateCommand command) {
        Budget budget = new Budget();
        budget.create(
            command.getBudgetCode(),
            command.getBudgetName(),
            Budget.BudgetType.valueOf(command.getType()),
            Budget.BudgetPeriod.valueOf(command.getPeriod()),
            command.getDepartmentId(),
            command.getDepartmentName(),
            command.getCreatedBy(),
            command.getCreatorName()
        );
        budget.setRemark(command.getRemark());
        
        Long budgetId = budgetRepository.save(budget);
        
        if (command.getLines() != null && !command.getLines().isEmpty()) {
            int sort = 0;
            for (BudgetLineCreateCommand lineCmd : command.getLines()) {
                BudgetLine line = new BudgetLine();
                line.create(budgetId, lineCmd.getSubjectCode(), lineCmd.getSubjectName(),
                           lineCmd.getBudgetAmount(), sort++);
                line.setRemark(lineCmd.getRemark());
                budgetRepository.saveLine(line);
            }
        }
        
        return budgetId;
    }

    @Transactional
    public void updateBudget(BudgetUpdateCommand command) {
        Budget budget = budgetRepository.findById(command.getId())
            .orElseThrow(() -> new IllegalArgumentException("Budget not found"));
        
        if (budget.getStatus() != Budget.BudgetStatus.DRAFT) {
            throw new IllegalStateException("Only draft budget can be updated");
        }
        
        budget.setBudgetName(command.getBudgetName());
        budget.setDepartmentId(command.getDepartmentId());
        budget.setDepartmentName(command.getDepartmentName());
        budget.setRemark(command.getRemark());
        
        budgetRepository.update(budget);
        
        if (command.getLines() != null) {
            List<BudgetLine> existingLines = budgetRepository.findLinesByBudgetId(budget.getId());
            for (BudgetLine line : existingLines) {
                budgetRepository.deleteLine(line.getId());
            }
            
            int sort = 0;
            for (BudgetLineCreateCommand lineCmd : command.getLines()) {
                BudgetLine line = new BudgetLine();
                line.create(budget.getId(), lineCmd.getSubjectCode(), lineCmd.getSubjectName(),
                           lineCmd.getBudgetAmount(), sort++);
                line.setRemark(lineCmd.getRemark());
                budgetRepository.saveLine(line);
            }
        }
    }

    @Transactional
    public void submitBudget(Long budgetId) {
        Budget budget = budgetRepository.findById(budgetId)
            .orElseThrow(() -> new IllegalArgumentException("Budget not found"));
        budget.submitForApproval();
        budgetRepository.update(budget);
    }

    @Transactional
    public void approveBudget(Long budgetId, Long approverId, String approverName) {
        Budget budget = budgetRepository.findById(budgetId)
            .orElseThrow(() -> new IllegalArgumentException("Budget not found"));
        budget.approve(approverId, approverName);
        budgetRepository.update(budget);
    }

    @Transactional
    public void rejectBudget(Long budgetId) {
        Budget budget = budgetRepository.findById(budgetId)
            .orElseThrow(() -> new IllegalArgumentException("Budget not found"));
        budget.reject();
        budgetRepository.update(budget);
    }

    @Transactional
    public void closeBudget(Long budgetId) {
        Budget budget = budgetRepository.findById(budgetId)
            .orElseThrow(() -> new IllegalArgumentException("Budget not found"));
        budget.close();
        budgetRepository.update(budget);
    }

    @Transactional
    public Long applyAdjustment(BudgetAdjustmentCreateCommand command) {
        Budget budget = budgetRepository.findById(command.getBudgetId())
            .orElseThrow(() -> new IllegalArgumentException("Budget not found"));
        
        if (budget.getStatus() != Budget.BudgetStatus.APPROVED) {
            throw new IllegalStateException("Only approved budget can be adjusted");
        }
        
        BudgetAdjustment adjustment = new BudgetAdjustment();
        adjustment.apply(
            budget.getId(),
            budget.getBudgetCode(),
            BudgetAdjustment.AdjustmentType.valueOf(command.getType()),
            command.getSubjectCode(),
            command.getSubjectName(),
            command.getOriginalAmount(),
            command.getAdjustedAmount(),
            command.getApplicantId(),
            command.getApplicantName(),
            command.getReason()
        );
        
        return budgetRepository.saveAdjustment(adjustment);
    }

    @Transactional
    public void approveAdjustment(Long adjustmentId, Long approverId, String approverName, String comment) {
        BudgetAdjustment adjustment = budgetRepository.findAdjustmentsByBudgetId(adjustmentId).stream()
            .filter(a -> a.getId().equals(adjustmentId))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("Adjustment not found"));
        
        adjustment.approve(approverId, approverName, comment);
        budgetRepository.updateAdjustment(adjustment);
    }

    @Transactional
    public void rejectAdjustment(Long adjustmentId, Long approverId, String approverName, String comment) {
        BudgetAdjustment adjustment = budgetRepository.findAdjustmentsByBudgetId(adjustmentId).stream()
            .filter(a -> a.getId().equals(adjustmentId))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("Adjustment not found"));
        
        adjustment.reject(approverId, approverName, comment);
        budgetRepository.updateAdjustment(adjustment);
    }

    @Transactional
    public void recordUsage(Long budgetId, String subjectCode, BigDecimal amount) {
        Budget budget = budgetRepository.findById(budgetId)
            .orElseThrow(() -> new IllegalArgumentException("Budget not found"));
        
        if (budget.getStatus() != Budget.BudgetStatus.APPROVED) {
            throw new IllegalStateException("Only approved budget can record usage");
        }
        
        budget.recordUsage(amount);
        budgetRepository.update(budget);
        
        List<BudgetLine> lines = budgetRepository.findLinesByBudgetId(budgetId);
        for (BudgetLine line : lines) {
            if (line.getSubjectCode().equals(subjectCode)) {
                line.recordUsage(amount);
                break;
            }
        }
    }

    @Transactional
    public void deleteBudget(Long budgetId) {
        Budget budget = budgetRepository.findById(budgetId)
            .orElseThrow(() -> new IllegalArgumentException("Budget not found"));
        
        if (budget.getStatus() != Budget.BudgetStatus.DRAFT && 
            budget.getStatus() != Budget.BudgetStatus.REJECTED) {
            throw new IllegalStateException("Only draft or rejected budget can be deleted");
        }
        
        List<BudgetLine> lines = budgetRepository.findLinesByBudgetId(budgetId);
        for (BudgetLine line : lines) {
            budgetRepository.deleteLine(line.getId());
        }
        
        budgetRepository.delete(budgetId);
    }
}