package com.metawebthree.finance.application.query.ledger;

import com.metawebthree.finance.domain.entity.ledger.GeneralLedger;
import com.metawebthree.finance.domain.repository.ledger.GeneralLedgerRepository;
import com.metawebthree.finance.domain.service.ledger.GeneralLedgerDomainService;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Service
public class LedgerQueryService {
    private final GeneralLedgerRepository ledgerRepository;
    private final GeneralLedgerDomainService ledgerDomainService;

    public LedgerQueryService(GeneralLedgerRepository ledgerRepository,
                              GeneralLedgerDomainService ledgerDomainService) {
        this.ledgerRepository = ledgerRepository;
        this.ledgerDomainService = ledgerDomainService;
    }

    public Optional<GeneralLedger> getLedgerById(Long id) {
        return ledgerRepository.findById(id);
    }

    public Optional<GeneralLedger> getLedgerByNo(String ledgerNo) {
        return ledgerRepository.findByLedgerNo(ledgerNo);
    }

    public Optional<GeneralLedger> getLedgerByPeriod(Integer periodYear, Integer periodMonth) {
        return ledgerRepository.findByPeriod(periodYear, periodMonth);
    }

    public List<GeneralLedger> getLedgersByStatus(GeneralLedger.LedgerStatus status) {
        return ledgerRepository.findByStatus(status);
    }

    public List<GeneralLedger> getLedgersByPeriodBetween(Integer startYear, Integer startMonth,
                                                          Integer endYear, Integer endMonth) {
        return ledgerRepository.findByPeriodBetween(startYear, startMonth, endYear, endMonth);
    }

    public Map<String, Object> getSubjectBalance(String subjectCode, Integer periodYear, Integer periodMonth) {
        Map<String, BigDecimal> balance = ledgerDomainService.getSubjectBalance(subjectCode, periodYear, periodMonth);

        Map<String, Object> result = new HashMap<>();
        result.put("subjectCode", subjectCode);
        result.put("periodYear", periodYear);
        result.put("periodMonth", periodMonth);
        result.put("totalDebit", balance.get("totalDebit"));
        result.put("totalCredit", balance.get("totalCredit"));
        result.put("balance", balance.get("balance"));

        return result;
    }

    public List<GeneralLedger.GeneralLedgerEntry> getEntriesBySubjectId(Long subjectId) {
        return ledgerRepository.findEntriesBySubjectId(subjectId);
    }

    public List<GeneralLedger.GeneralLedgerEntry> getEntriesBySubjectCode(String subjectCode) {
        return ledgerRepository.findEntriesBySubjectCode(subjectCode);
    }
}