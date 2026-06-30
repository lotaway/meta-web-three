package com.metawebthree.finance.application.command.cash;

import com.metawebthree.finance.application.command.cash.dto.*;
import com.metawebthree.finance.domain.entity.cash.*;
import com.metawebthree.finance.domain.repository.cash.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.UUID;

@Service
public class CashCommandService {
    private final CashPlanRepository cashPlanRepository;
    private final BankAccountRepository bankAccountRepository;
    private final BankReconciliationRepository bankReconciliationRepository;
    private final CashTransferRepository cashTransferRepository;
    private final CashFlowForecastRepository cashFlowForecastRepository;

    public CashCommandService(CashPlanRepository cashPlanRepository,
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

    // Cash Plan operations
    @Transactional
    public Long createCashPlan(CashPlanCreateCommand command) {
        CashPlan cashPlan = new CashPlan();
        cashPlan.create(
            command.getPlanCode(),
            command.getPlanName(),
            CashPlan.CashPlanType.valueOf(command.getType()),
            CashPlan.CashPlanPeriod.valueOf(command.getPeriod()),
            command.getStartDate(),
            command.getEndDate(),
            command.getDepartmentId(),
            command.getDepartmentName(),
            command.getCreatedBy(),
            command.getCreatorName()
        );
        cashPlan.setRemark(command.getRemark());
        
        Long planId = cashPlanRepository.save(cashPlan);
        
        if (command.getLines() != null && !command.getLines().isEmpty()) {
            int sort = 0;
            for (CashPlanCreateCommand.CashPlanLineCreateCommand lineCmd : command.getLines()) {
                CashPlanLine line = new CashPlanLine();
                line.create(planId, lineCmd.getCategoryCode(), lineCmd.getCategoryName(),
                           CashPlanLine.CashFlowDirection.valueOf(lineCmd.getFlowDirection()),
                           lineCmd.getPlannedAmount(), lineCmd.getPlannedDate(), sort++);
                line.setRemark(lineCmd.getRemark());
                cashPlanRepository.saveLine(line);
            }
        }
        
        return planId;
    }

    @Transactional
    public void updateCashPlan(CashPlanCreateCommand command) {
        CashPlan cashPlan = cashPlanRepository.findByPlanCode(command.getPlanCode())
            .orElseThrow(() -> new IllegalArgumentException("Cash plan not found"));
        
        if (cashPlan.getStatus() != CashPlan.CashPlanStatus.DRAFT) {
            throw new IllegalStateException("Only draft cash plan can be updated");
        }
        
        cashPlan.setPlanName(command.getPlanName());
        cashPlan.setStartDate(command.getStartDate());
        cashPlan.setEndDate(command.getEndDate());
        cashPlan.setRemark(command.getRemark());
        
        cashPlanRepository.update(cashPlan);
    }

    @Transactional
    public void submitCashPlan(Long id) {
        CashPlan cashPlan = cashPlanRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Cash plan not found"));
        cashPlan.submitForApproval();
        cashPlanRepository.update(cashPlan);
    }

    @Transactional
    public void approveCashPlan(Long id, Long approverId, String approverName) {
        CashPlan cashPlan = cashPlanRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Cash plan not found"));
        cashPlan.approve(approverId, approverName);
        cashPlanRepository.update(cashPlan);
    }

    @Transactional
    public void rejectCashPlan(Long id) {
        CashPlan cashPlan = cashPlanRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Cash plan not found"));
        cashPlan.reject();
        cashPlanRepository.update(cashPlan);
    }

    @Transactional
    public void deleteCashPlan(Long id) {
        cashPlanRepository.deleteById(id);
    }

    // Bank Account operations
    @Transactional
    public Long createBankAccount(BankAccountCreateCommand command) {
        BankAccount account = new BankAccount();
        account.create(
            command.getAccountCode(),
            command.getAccountName(),
            command.getBankName(),
            command.getAccountNumber(),
            command.getAccountType(),
            command.getCurrency() != null ? command.getCurrency() : "CNY",
            command.getCreatedBy(),
            command.getCreatorName()
        );
        account.setRemark(command.getRemark());
        return bankAccountRepository.save(account);
    }

    @Transactional
    public void freezeBankAccount(Long id) {
        BankAccount account = bankAccountRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Bank account not found"));
        account.freeze();
        bankAccountRepository.update(account);
    }

    @Transactional
    public void unfreezeBankAccount(Long id) {
        BankAccount account = bankAccountRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Bank account not found"));
        account.unfreeze();
        bankAccountRepository.update(account);
    }

    @Transactional
    public void closeBankAccount(Long id) {
        BankAccount account = bankAccountRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Bank account not found"));
        account.close();
        bankAccountRepository.update(account);
    }

    @Transactional
    public void deleteBankAccount(Long id) {
        bankAccountRepository.deleteById(id);
    }

    // Cash Transfer operations
    @Transactional
    public Long createCashTransfer(CashTransferCreateCommand command) {
        CashTransfer transfer = new CashTransfer();
        transfer.create(
            command.getTransferNo() != null ? command.getTransferNo() : generateTransferNo(),
            command.getFromAccountId(),
            command.getFromAccountName(),
            command.getToAccountId(),
            command.getToAccountName(),
            command.getAmount(),
            command.getCurrency() != null ? command.getCurrency() : "CNY",
            CashTransfer.CashTransferType.valueOf(command.getType()),
            command.getPurpose(),
            command.getCreatedBy(),
            command.getCreatorName()
        );
        transfer.setRemark(command.getRemark());
        return cashTransferRepository.save(transfer);
    }

    @Transactional
    public void submitCashTransfer(Long id) {
        CashTransfer transfer = cashTransferRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Cash transfer not found"));
        transfer.submitForApproval();
        cashTransferRepository.update(transfer);
    }

    @Transactional
    public void approveCashTransfer(Long id, Long approverId, String approverName) {
        CashTransfer transfer = cashTransferRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Cash transfer not found"));
        transfer.approve(approverId, approverName);
        
        // Execute the transfer
        BankAccount fromAccount = bankAccountRepository.findById(transfer.getFromAccountId())
            .orElseThrow(() -> new IllegalArgumentException("From account not found"));
        BankAccount toAccount = bankAccountRepository.findById(transfer.getToAccountId())
            .orElseThrow(() -> new IllegalArgumentException("To account not found"));
        
        fromAccount.withdraw(transfer.getAmount());
        toAccount.deposit(transfer.getAmount());
        
        bankAccountRepository.update(fromAccount);
        bankAccountRepository.update(toAccount);
        transfer.execute("System");
        
        cashTransferRepository.update(transfer);
    }

    @Transactional
    public void rejectCashTransfer(Long id) {
        CashTransfer transfer = cashTransferRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Cash transfer not found"));
        transfer.reject();
        cashTransferRepository.update(transfer);
    }

    @Transactional
    public void cancelCashTransfer(Long id) {
        CashTransfer transfer = cashTransferRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Cash transfer not found"));
        transfer.cancel();
        cashTransferRepository.update(transfer);
    }

    @Transactional
    public void deleteCashTransfer(Long id) {
        cashTransferRepository.deleteById(id);
    }

    // Bank Reconciliation operations
    @Transactional
    public Long createBankReconciliation(BankReconciliationCreateCommand command) {
        BankReconciliation reconciliation = new BankReconciliation();
        reconciliation.create(
            command.getReconciliationNo() != null ? command.getReconciliationNo() : generateReconciliationNo(),
            command.getBankAccountId(),
            command.getBankAccountName(),
            command.getBankName(),
            command.getStatementDate(),
            command.getStatementEndDate(),
            command.getBankBalance(),
            command.getBookBalance(),
            command.getCreatedBy(),
            command.getCreatorName()
        );
        reconciliation.setRemark(command.getRemark());
        return bankReconciliationRepository.save(reconciliation);
    }

    @Transactional
    public void submitBankReconciliation(Long id) {
        BankReconciliation reconciliation = bankReconciliationRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Bank reconciliation not found"));
        reconciliation.submitForApproval();
        bankReconciliationRepository.update(reconciliation);
    }

    @Transactional
    public void approveBankReconciliation(Long id, Long approverId, String approverName) {
        BankReconciliation reconciliation = bankReconciliationRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Bank reconciliation not found"));
        reconciliation.approve(approverId, approverName);
        bankReconciliationRepository.update(reconciliation);
    }

    @Transactional
    public void deleteBankReconciliation(Long id) {
        bankReconciliationRepository.deleteById(id);
    }

    // Cash Flow Forecast operations
    @Transactional
    public Long createCashFlowForecast(CashFlowForecastCreateCommand command) {
        CashFlowForecast forecast = new CashFlowForecast();
        forecast.create(
            command.getForecastNo() != null ? command.getForecastNo() : generateForecastNo(),
            command.getForecastDate(),
            command.getStartDate(),
            command.getEndDate(),
            command.getCurrency() != null ? command.getCurrency() : "CNY",
            command.getOpeningBalance(),
            command.getCreatedBy(),
            command.getCreatorName()
        );
        forecast.setRemark(command.getRemark());
        
        Long forecastId = cashFlowForecastRepository.save(forecast);
        
        if (command.getItems() != null && !command.getItems().isEmpty()) {
            for (CashFlowForecastCreateCommand.ForecastItemCreateCommand itemCmd : command.getItems()) {
                CashFlowForecast.ForecastItem item = new CashFlowForecast.ForecastItem();
                item.setForecastId(forecastId);
                item.setCategoryCode(itemCmd.getCategoryCode());
                item.setCategoryName(itemCmd.getCategoryName());
                item.setFlowDirection(CashFlowForecast.ForecastItem.FlowDirection.valueOf(itemCmd.getFlowDirection()));
                item.setAmount(itemCmd.getAmount());
                item.setPredictedDate(itemCmd.getPredictedDate());
                item.setDescription(itemCmd.getDescription());
                item.setConfidenceLevel(itemCmd.getConfidenceLevel());
                item.setRemark(itemCmd.getRemark());
                cashFlowForecastRepository.saveItem(item);
            }
        }
        
        return forecastId;
    }

    @Transactional
    public void deleteCashFlowForecast(Long id) {
        cashFlowForecastRepository.deleteById(id);
    }

    private String generateTransferNo() {
        return "TRF-" + LocalDateTime.now().toString().replace("-", "").replace(":", "").substring(0, 14) + "-" + UUID.randomUUID().toString().substring(0, 4).toUpperCase();
    }

    private String generateReconciliationNo() {
        return "REC-" + LocalDateTime.now().toString().replace("-", "").replace(":", "").substring(0, 14) + "-" + UUID.randomUUID().toString().substring(0, 4).toUpperCase();
    }

    private String generateForecastNo() {
        return "FCF-" + LocalDateTime.now().toString().replace("-", "").replace(":", "").substring(0, 14) + "-" + UUID.randomUUID().toString().substring(0, 4).toUpperCase();
    }
}