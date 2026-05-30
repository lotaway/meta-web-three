package com.metawebthree.finance.application.query.budget;

import com.metawebthree.finance.domain.entity.budget.Budget;
import com.metawebthree.finance.domain.entity.budget.BudgetAdjustment;
import com.metawebthree.finance.domain.entity.budget.BudgetLine;
import com.metawebthree.finance.domain.repository.budget.BudgetRepository;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class BudgetQueryService {
    private final BudgetRepository budgetRepository;

    public BudgetQueryService(BudgetRepository budgetRepository) {
        this.budgetRepository = budgetRepository;
    }

    public Budget getById(Long id) {
        Budget budget = budgetRepository.findById(id).orElse(null);
        if (budget != null) {
            List<BudgetLine> lines = budgetRepository.findLinesByBudgetId(id);
            budget.setLines(lines);
        }
        return budget;
    }

    public Budget getByCode(String budgetCode) {
        Budget budget = budgetRepository.findByCode(budgetCode).orElse(null);
        if (budget != null) {
            List<BudgetLine> lines = budgetRepository.findLinesByBudgetId(budget.getId());
            budget.setLines(lines);
        }
        return budget;
    }

    public List<Budget> listByDepartment(Long departmentId) {
        return budgetRepository.findByDepartmentId(departmentId).stream()
            .peek(budget -> {
                List<BudgetLine> lines = budgetRepository.findLinesByBudgetId(budget.getId());
                budget.setLines(lines);
            })
            .collect(Collectors.toList());
    }

    public List<Budget> listByStatus(String status) {
        return budgetRepository.findByStatus(Budget.BudgetStatus.valueOf(status)).stream()
            .peek(budget -> {
                List<BudgetLine> lines = budgetRepository.findLinesByBudgetId(budget.getId());
                budget.setLines(lines);
            })
            .collect(Collectors.toList());
    }

    public List<Budget> listByPeriod(String period) {
        return budgetRepository.findByPeriod(Budget.BudgetPeriod.valueOf(period)).stream()
            .peek(budget -> {
                List<BudgetLine> lines = budgetRepository.findLinesByBudgetId(budget.getId());
                budget.setLines(lines);
            })
            .collect(Collectors.toList());
    }

    public List<Budget> listAll() {
        return budgetRepository.findByStatus(Budget.BudgetStatus.APPROVED).stream()
            .peek(budget -> {
                List<BudgetLine> lines = budgetRepository.findLinesByBudgetId(budget.getId());
                budget.setLines(lines);
            })
            .collect(Collectors.toList());
    }

    public List<BudgetLine> getBudgetLines(Long budgetId) {
        return budgetRepository.findLinesByBudgetId(budgetId);
    }

    public List<BudgetAdjustment> getAdjustments(Long budgetId) {
        return budgetRepository.findAdjustmentsByBudgetId(budgetId);
    }

    public List<BudgetAdjustment> getPendingAdjustments() {
        return budgetRepository.findPendingAdjustments();
    }

    public BudgetAnalysisResult analyzeBudget(Long budgetId) {
        Budget budget = getById(budgetId);
        if (budget == null) {
            return null;
        }
        
        BudgetAnalysisResult result = new BudgetAnalysisResult();
        result.setBudgetId(budget.getId());
        result.setBudgetCode(budget.getBudgetCode());
        result.setBudgetName(budget.getBudgetName());
        result.setTotalBudget(budget.getTotalAmount());
        result.setAdjustedBudget(budget.getTotalAmount().add(budget.getAdjustedAmount()));
        result.setUsedAmount(budget.getUsedAmount());
        result.setAvailableAmount(budget.getAvailableAmount());
        result.setUsageRate(budget.getUsageRate());
        
        if (budget.getLines() != null && !budget.getLines().isEmpty()) {
            List<BudgetLineAnalysis> lineAnalyses = budget.getLines().stream()
                .map(line -> {
                    BudgetLineAnalysis analysis = new BudgetLineAnalysis();
                    analysis.setSubjectCode(line.getSubjectCode());
                    analysis.setSubjectName(line.getSubjectName());
                    analysis.setBudgetAmount(line.getBudgetAmount());
                    analysis.setAdjustedAmount(line.getAdjustedAmount());
                    analysis.setUsedAmount(line.getUsedAmount());
                    analysis.setAvailableAmount(line.getAvailableAmount());
                    analysis.setUsageRate(line.getUsageRate());
                    return analysis;
                })
                .collect(Collectors.toList());
            result.setLineAnalyses(lineAnalyses);
        }
        
        return result;
    }

    public BudgetComparisonResult compareBudgetWithActual(Long budgetId, BigDecimal actualAmount) {
        Budget budget = getById(budgetId);
        if (budget == null) {
            return null;
        }
        
        BudgetComparisonResult result = new BudgetComparisonResult();
        result.setBudgetId(budget.getId());
        result.setBudgetCode(budget.getBudgetCode());
        result.setBudgetAmount(budget.getTotalAmount().add(budget.getAdjustedAmount()));
        result.setActualAmount(actualAmount);
        result.setVariance(actualAmount.subtract(result.getBudgetAmount()));
        
        BigDecimal budgetAmt = result.getBudgetAmount();
        if (budgetAmt.compareTo(BigDecimal.ZERO) != 0) {
            result.setVarianceRate(result.getVariance()
                .divide(budgetAmt, 4, BigDecimal.ROUND_HALF_UP)
                .multiply(BigDecimal.valueOf(100)));
        } else {
            result.setVarianceRate(BigDecimal.ZERO);
        }
        
        result.setWithinBudget(result.getVariance().compareTo(BigDecimal.ZERO) <= 0);
        
        return result;
    }

    public static class BudgetAnalysisResult {
        private Long budgetId;
        private String budgetCode;
        private String budgetName;
        private BigDecimal totalBudget;
        private BigDecimal adjustedBudget;
        private BigDecimal usedAmount;
        private BigDecimal availableAmount;
        private BigDecimal usageRate;
        private List<BudgetLineAnalysis> lineAnalyses;

        public Long getBudgetId() { return budgetId; }
        public String getBudgetCode() { return budgetCode; }
        public String getBudgetName() { return budgetName; }
        public BigDecimal getTotalBudget() { return totalBudget; }
        public BigDecimal getAdjustedBudget() { return adjustedBudget; }
        public BigDecimal getUsedAmount() { return usedAmount; }
        public BigDecimal getAvailableAmount() { return availableAmount; }
        public BigDecimal getUsageRate() { return usageRate; }
        public List<BudgetLineAnalysis> getLineAnalyses() { return lineAnalyses; }

        public void setBudgetId(Long budgetId) { this.budgetId = budgetId; }
        public void setBudgetCode(String budgetCode) { this.budgetCode = budgetCode; }
        public void setBudgetName(String budgetName) { this.budgetName = budgetName; }
        public void setTotalBudget(BigDecimal totalBudget) { this.totalBudget = totalBudget; }
        public void setAdjustedBudget(BigDecimal adjustedBudget) { this.adjustedBudget = adjustedBudget; }
        public void setUsedAmount(BigDecimal usedAmount) { this.usedAmount = usedAmount; }
        public void setAvailableAmount(BigDecimal availableAmount) { this.availableAmount = availableAmount; }
        public void setUsageRate(BigDecimal usageRate) { this.usageRate = usageRate; }
        public void setLineAnalyses(List<BudgetLineAnalysis> lineAnalyses) { this.lineAnalyses = lineAnalyses; }
    }

    public static class BudgetLineAnalysis {
        private String subjectCode;
        private String subjectName;
        private BigDecimal budgetAmount;
        private BigDecimal adjustedAmount;
        private BigDecimal usedAmount;
        private BigDecimal availableAmount;
        private BigDecimal usageRate;

        public String getSubjectCode() { return subjectCode; }
        public String getSubjectName() { return subjectName; }
        public BigDecimal getBudgetAmount() { return budgetAmount; }
        public BigDecimal getAdjustedAmount() { return adjustedAmount; }
        public BigDecimal getUsedAmount() { return usedAmount; }
        public BigDecimal getAvailableAmount() { return availableAmount; }
        public BigDecimal getUsageRate() { return usageRate; }

        public void setSubjectCode(String subjectCode) { this.subjectCode = subjectCode; }
        public void setSubjectName(String subjectName) { this.subjectName = subjectName; }
        public void setBudgetAmount(BigDecimal budgetAmount) { this.budgetAmount = budgetAmount; }
        public void setAdjustedAmount(BigDecimal adjustedAmount) { this.adjustedAmount = adjustedAmount; }
        public void setUsedAmount(BigDecimal usedAmount) { this.usedAmount = usedAmount; }
        public void setAvailableAmount(BigDecimal availableAmount) { this.availableAmount = availableAmount; }
        public void setUsageRate(BigDecimal usageRate) { this.usageRate = usageRate; }
    }

    public static class BudgetComparisonResult {
        private Long budgetId;
        private String budgetCode;
        private BigDecimal budgetAmount;
        private BigDecimal actualAmount;
        private BigDecimal variance;
        private BigDecimal varianceRate;
        private Boolean withinBudget;

        public Long getBudgetId() { return budgetId; }
        public String getBudgetCode() { return budgetCode; }
        public BigDecimal getBudgetAmount() { return budgetAmount; }
        public BigDecimal getActualAmount() { return actualAmount; }
        public BigDecimal getVariance() { return variance; }
        public BigDecimal getVarianceRate() { return varianceRate; }
        public Boolean getWithinBudget() { return withinBudget; }

        public void setBudgetId(Long budgetId) { this.budgetId = budgetId; }
        public void setBudgetCode(String budgetCode) { this.budgetCode = budgetCode; }
        public void setBudgetAmount(BigDecimal budgetAmount) { this.budgetAmount = budgetAmount; }
        public void setActualAmount(BigDecimal actualAmount) { this.actualAmount = actualAmount; }
        public void setVariance(BigDecimal variance) { this.variance = variance; }
        public void setVarianceRate(BigDecimal varianceRate) { this.varianceRate = varianceRate; }
        public void setWithinBudget(Boolean withinBudget) { this.withinBudget = withinBudget; }
    }
}