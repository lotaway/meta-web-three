package com.metawebthree.finance.application.query.arap;

import com.metawebthree.finance.application.query.arap.dto.ArQueryResult;
import com.metawebthree.finance.application.query.arap.dto.ApQueryResult;
import com.metawebthree.finance.domain.entity.arap.AccountsReceivable.ArStatus;
import com.metawebthree.finance.domain.entity.arap.AccountsPayable.ApStatus;
import com.metawebthree.finance.domain.repository.arap.AccountsReceivableRepository;
import com.metawebthree.finance.domain.repository.arap.AccountsPayableRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDate;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class ArApQueryService {
    private final AccountsReceivableRepository arRepository;
    private final AccountsPayableRepository apRepository;

    public ArApQueryService(AccountsReceivableRepository arRepository,
                            AccountsPayableRepository apRepository) {
        this.arRepository = arRepository;
        this.apRepository = apRepository;
    }

    public ArQueryResult getArById(Long id) {
        return arRepository.findById(id)
            .map(ArQueryResult::fromEntity)
            .orElse(null);
    }

    public ArQueryResult getArByCode(String arCode) {
        return arRepository.findByArCode(arCode)
            .map(ArQueryResult::fromEntity)
            .orElse(null);
    }

    public List<ArQueryResult> getArByCustomerId(Long customerId) {
        return arRepository.findByCustomerId(customerId).stream()
            .map(ArQueryResult::fromEntity)
            .collect(Collectors.toList());
    }

    public List<ArQueryResult> getArByStatus(String status) {
        ArStatus arStatus = ArStatus.valueOf(status);
        return arRepository.findByStatus(arStatus).stream()
            .map(ArQueryResult::fromEntity)
            .collect(Collectors.toList());
    }

    public List<ArQueryResult> getOverdueAr() {
        return arRepository.findByDueDateBefore(LocalDate.now()).stream()
            .map(ArQueryResult::fromEntity)
            .collect(Collectors.toList());
    }

    public List<ArQueryResult> getAllAr() {
        return arRepository.findAll().stream()
            .map(ArQueryResult::fromEntity)
            .collect(Collectors.toList());
    }

    public ApQueryResult getApById(Long id) {
        return apRepository.findById(id)
            .map(ApQueryResult::fromEntity)
            .orElse(null);
    }

    public ApQueryResult getApByCode(String apCode) {
        return apRepository.findByApCode(apCode)
            .map(ApQueryResult::fromEntity)
            .orElse(null);
    }

    public List<ApQueryResult> getApBySupplierId(Long supplierId) {
        return apRepository.findBySupplierId(supplierId).stream()
            .map(ApQueryResult::fromEntity)
            .collect(Collectors.toList());
    }

    public List<ApQueryResult> getApByStatus(String status) {
        ApStatus apStatus = ApStatus.valueOf(status);
        return apRepository.findByStatus(apStatus).stream()
            .map(ApQueryResult::fromEntity)
            .collect(Collectors.toList());
    }

    public List<ApQueryResult> getOverdueAp() {
        return apRepository.findByDueDateBefore(LocalDate.now()).stream()
            .map(ApQueryResult::fromEntity)
            .collect(Collectors.toList());
    }

    public List<ApQueryResult> getAllAp() {
        return apRepository.findAll().stream()
            .map(ApQueryResult::fromEntity)
            .collect(Collectors.toList());
    }
}