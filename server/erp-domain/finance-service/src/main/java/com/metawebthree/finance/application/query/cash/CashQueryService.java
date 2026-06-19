package com.metawebthree.finance.application.query.cash;

import com.metawebthree.finance.domain.entity.cash.*;
import com.metawebthree.finance.domain.repository.cash.*;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

@Service
public class CashQueryService {
    private final CashPlanRepository cashPlanRepository;
    private final BankAccountRepository bankAccountRepository;
    private final BankReconciliationRepository bankReconciliationRepository;
    private final CashTransferRepository cashTransferRepository;
    private final CashFlowForecastRepository cashFlowForecastRepository;

    public CashQueryService(CashPlanRepository cashPlanRepository,
                           BankAccountRepository bankAccountRepository,
                           BankReconciliationRepository bankReconciliationRepository,
                           CashTransferRepository cashTransferRepository,
                           CashFlowForecastRepository cashFlowForecastRepository) {
        this.cashPlanRepository = cashPlanRepository;
        this.bankAccountRepository = bankAccountRepository;
        this.bankReconciliationRepository = bankReconciliationRepository;
        this.cashTransferRepository = cashTransferRepository;
        this.cashFlowForecastRepository = cashFlowForecastRepository;
    }

    // Cash Plan queries
    public CashPlan getCashPlanById(Long id) {
        return cashPlanRepository.findById(id).orElse(null);
    }

    public CashPlan getCashPlanByCode(String planCode) {
        return cashPlanRepository.findByPlanCode(planCode).orElse(null);
    }

    public List<CashPlan> listAllCashPlans() {
        return cashPlanRepository.findAll();
    }

    public List<CashPlan> listCashPlansByStatus(String status) {
        return cashPlanRepository.findByStatus(CashPlan.CashPlanStatus.valueOf(status));
    }

    public List<CashPlan> listCashPlansByDepartment(Long departmentId) {
        return cashPlanRepository.findByDepartmentId(departmentId);
    }

    public List<CashPlanLine> getCashPlanLines(Long cashPlanId) {
        return cashPlanRepository.findLinesByCashPlanId(cashPlanId);
    }

    // Bank Account queries
    public BankAccount getBankAccountById(Long id) {
        return bankAccountRepository.findById(id).orElse(null);
    }

    public BankAccount getBankAccountByCode(String accountCode) {
        return bankAccountRepository.findByAccountCode(accountCode).orElse(null);
    }

    public List<BankAccount> listAllBankAccounts() {
        return bankAccountRepository.findAll();
    }

    public List<BankAccount> listBankAccountsByStatus(String status) {
        return bankAccountRepository.findByStatus(BankAccount.BankAccountStatus.valueOf(status));
    }

    public List<BankAccount> listActiveBankAccounts() {
        return bankAccountRepository.findByIsActive(true);
    }

    public BigDecimal getTotalCashBalance() {
        return bankAccountRepository.findByIsActive(true).stream()
            .map(BankAccount::getBalance)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    // Cash Transfer queries
    public CashTransfer getCashTransferById(Long id) {
        return cashTransferRepository.findById(id).orElse(null);
    }

    public List<CashTransfer> listAllCashTransfers() {
        return cashTransferRepository.findAll();
    }

    public List<CashTransfer> listCashTransfersByStatus(String status) {
        return cashTransferRepository.findByStatus(CashTransfer.CashTransferStatus.valueOf(status));
    }

    public List<CashTransfer> listCashTransfersByAccount(Long accountId) {
        return cashTransferRepository.findByFromAccountIdOrToAccountId(accountId, accountId);
    }

    // Bank Reconciliation queries
    public BankReconciliation getBankReconciliationById(Long id) {
        return bankReconciliationRepository.findById(id).orElse(null);
    }

    public List<BankReconciliation> listAllBankReconciliations() {
        return bankReconciliationRepository.findAll();
    }

    public List<BankReconciliation> listBankReconciliationsByAccount(Long accountId) {
        return bankReconciliationRepository.findByBankAccountId(accountId);
    }

    public List<BankReconciliation> listBankReconciliationsByStatus(String status) {
        return bankReconciliationRepository.findByStatus(BankReconciliation.ReconciliationStatus.valueOf(status));
    }

    // Cash Flow Forecast queries
    public CashFlowForecast getCashFlowForecastById(Long id) {
        return cashFlowForecastRepository.findById(id).orElse(null);
    }

    public List<CashFlowForecast> listAllCashFlowForecasts() {
        return cashFlowForecastRepository.findAll();
    }

    public List<CashFlowForecast> listCashFlowForecastsByDate(LocalDate forecastDate) {
        return cashFlowForecastRepository.findByForecastDate(forecastDate);
    }

    public List<CashFlowForecast.ForecastItem> getCashFlowForecastItems(Long forecastId) {
        return cashFlowForecastRepository.findItemsByForecastId(forecastId);
    }

    // Dashboard analytics
    public CashSummary getCashSummary() {
        CashSummary summary = new CashSummary();
        
        // Total balance across all active accounts
        summary.setTotalBalance(getTotalCashBalance());
        
        // Active bank accounts count
        summary.setActiveAccountCount((long) listActiveBankAccounts().size());
        
        // Pending transfers count
        long pendingTransfers = listCashTransfersByStatus("PENDING_APPROVAL").size();
        summary.setPendingTransferCount(pendingTransfers);
        
        // Pending reconciliations count
        long pendingReconciliations = listBankReconciliationsByStatus("PENDING_APPROVAL").size();
        summary.setPendingReconciliationCount(pendingReconciliations);
        
        // Draft cash plans count
        long draftPlans = listCashPlansByStatus("DRAFT").size();
        summary.setDraftPlanCount(draftPlans);
        
        return summary;
    }

    public static class CashSummary {
        private BigDecimal totalBalance;
        private Long activeAccountCount;
        private Long pendingTransferCount;
        private Long pendingReconciliationCount;
        private Long draftPlanCount;

        public BigDecimal getTotalBalance() { return totalBalance; }
        public void setTotalBalance(BigDecimal totalBalance) { this.totalBalance = totalBalance; }
        public Long getActiveAccountCount() { return activeAccountCount; }
        public void setActiveAccountCount(Long activeAccountCount) { this.activeAccountCount = activeAccountCount; }
        public Long getPendingTransferCount() { return pendingTransferCount; }
        public void setPendingTransferCount(Long pendingTransferCount) { this.pendingTransferCount = pendingTransferCount; }
        public Long getPendingReconciliationCount() { return pendingReconciliationCount; }
        public void setPendingReconciliationCount(Long pendingReconciliationCount) { this.pendingReconciliationCount = pendingReconciliationCount; }
        public Long getDraftPlanCount() { return draftPlanCount; }
        public void setDraftPlanCount(Long draftPlanCount) { this.draftPlanCount = draftPlanCount; }
    }
}