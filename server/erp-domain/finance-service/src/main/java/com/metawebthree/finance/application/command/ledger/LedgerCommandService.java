package com.metawebthree.finance.application.command.ledger;

import com.metawebthree.finance.application.command.ledger.dto.LedgerGenerateCommand;
import com.metawebthree.finance.domain.entity.ledger.GeneralLedger;
import com.metawebthree.finance.domain.repository.VoucherRepository;
import com.metawebthree.finance.domain.repository.ledger.GeneralLedgerRepository;
import com.metawebthree.finance.domain.service.ledger.GeneralLedgerDomainService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class LedgerCommandService {
    private final GeneralLedgerRepository ledgerRepository;
    private final VoucherRepository voucherRepository;
    private final GeneralLedgerDomainService ledgerDomainService;

    public LedgerCommandService(GeneralLedgerRepository ledgerRepository,
                                VoucherRepository voucherRepository,
                                GeneralLedgerDomainService ledgerDomainService) {
        this.ledgerRepository = ledgerRepository;
        this.voucherRepository = voucherRepository;
        this.ledgerDomainService = ledgerDomainService;
    }

    @Transactional
    public GeneralLedger generateLedgerFromVoucher(LedgerGenerateCommand command) {
        var voucher = voucherRepository.findById(command.getVoucherId())
                .orElseThrow(() -> new IllegalArgumentException("Voucher not found"));

        GeneralLedger ledger = ledgerDomainService.generateLedgerFromVoucher(voucher);
        ledgerRepository.update(ledger);

        return ledger;
    }

    @Transactional
    public GeneralLedger postLedger(Long ledgerId) {
        var ledger = ledgerRepository.findById(ledgerId)
                .orElseThrow(() -> new IllegalArgumentException("Ledger not found"));

        ledger.post();
        ledgerRepository.update(ledger);

        return ledger;
    }

    @Transactional
    public GeneralLedger closeLedger(Long ledgerId) {
        var ledger = ledgerRepository.findById(ledgerId)
                .orElseThrow(() -> new IllegalArgumentException("Ledger not found"));

        ledger.close();
        ledgerRepository.update(ledger);

        return ledger;
    }
}