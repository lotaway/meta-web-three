package com.metawebthree.finance.application.query;

import com.metawebthree.finance.domain.entity.Voucher;
import com.metawebthree.finance.domain.repository.VoucherRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;

@Service
public class VoucherQueryService {
    private final VoucherRepository voucherRepository;

    public VoucherQueryService(VoucherRepository voucherRepository) {
        this.voucherRepository = voucherRepository;
    }

    public Optional<Voucher> getById(Long voucherId) {
        return voucherRepository.findById(voucherId);
    }

    public Optional<Voucher> getByVoucherNo(String voucherNo) {
        return voucherRepository.findByVoucherNo(voucherNo);
    }

    public List<Voucher> listByStatus(String status) {
        Voucher.VoucherStatus voucherStatus = Voucher.VoucherStatus.valueOf(status.toUpperCase());
        return voucherRepository.findByStatus(voucherStatus);
    }

    public List<Voucher> listAll() {
        return voucherRepository.findAll();
    }

    public List<Voucher> listByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end) {
        return voucherRepository.findByVoucherDateBetween(start, end);
    }
}