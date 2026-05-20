package com.metawebthree.finance.application.command;

import com.metawebthree.finance.domain.entity.Voucher;
import com.metawebthree.finance.domain.repository.VoucherRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;

@Service
public class VoucherCommandService {
    private final VoucherRepository voucherRepository;

    public VoucherCommandService(VoucherRepository voucherRepository) {
        this.voucherRepository = voucherRepository;
    }

    public Long createVoucher(String voucherNo, String type, String description, String createdBy) {
        Voucher.VoucherType voucherType = Voucher.VoucherType.valueOf(type.toUpperCase());
        Voucher voucher = new Voucher();
        voucher.createDraft(voucherNo, voucherType, description, createdBy);
        voucherRepository.save(voucher);
        return voucher.getId();
    }

    public void addVoucherLine(Long voucherId, Long subjectId, BigDecimal debitAmount, BigDecimal creditAmount) {
        Voucher voucher = voucherRepository.findById(voucherId)
            .orElseThrow(() -> new IllegalArgumentException("Voucher not found"));
        voucher.addLine(subjectId, debitAmount, creditAmount);
        voucherRepository.update(voucher);
    }

    public void submitForApproval(Long voucherId) {
        Voucher voucher = voucherRepository.findById(voucherId)
            .orElseThrow(() -> new IllegalArgumentException("Voucher not found"));
        voucher.submitForApproval();
        voucherRepository.update(voucher);
    }

    public void approve(Long voucherId, String approver) {
        Voucher voucher = voucherRepository.findById(voucherId)
            .orElseThrow(() -> new IllegalArgumentException("Voucher not found"));
        voucher.approve(approver);
        voucherRepository.update(voucher);
    }

    public void reject(Long voucherId, String approver, String reason) {
        Voucher voucher = voucherRepository.findById(voucherId)
            .orElseThrow(() -> new IllegalArgumentException("Voucher not found"));
        voucher.reject(approver, reason);
        voucherRepository.update(voucher);
    }

    public void post(Long voucherId) {
        Voucher voucher = voucherRepository.findById(voucherId)
            .orElseThrow(() -> new IllegalArgumentException("Voucher not found"));
        voucher.post();
        voucherRepository.update(voucher);
    }
}