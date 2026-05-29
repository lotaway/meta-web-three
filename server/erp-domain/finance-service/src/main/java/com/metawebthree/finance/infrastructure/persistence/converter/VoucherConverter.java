package com.metawebthree.finance.infrastructure.persistence.converter;

import com.metawebthree.finance.domain.entity.Voucher;
import com.metawebthree.finance.infrastructure.persistence.dataobject.VoucherDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.VoucherLineDO;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class VoucherConverter {

    public Voucher toEntity(VoucherDO doObj, List<VoucherLineDO> lineDOs) {
        if (doObj == null) {
            return null;
        }
        
        List<Voucher.VoucherLine> lines = null;
        if (lineDOs != null && !lineDOs.isEmpty()) {
            lines = new ArrayList<>();
            for (VoucherLineDO lineDO : lineDOs) {
                Voucher.VoucherLine line = Voucher.VoucherLine.builder()
                        .subjectId(lineDO.getSubjectId())
                        .debitAmount(lineDO.getDebitAmount())
                        .creditAmount(lineDO.getCreditAmount())
                        .build();
                lines.add(line);
            }
        }
        
        return Voucher.builder()
                .id(doObj.getId())
                .voucherNo(doObj.getVoucherNo())
                .type(Voucher.VoucherType.valueOf(doObj.getType()))
                .voucherDate(doObj.getVoucherDate())
                .description(doObj.getDescription())
                .status(Voucher.VoucherStatus.valueOf(doObj.getStatus()))
                .createdBy(doObj.getCreatedBy())
                .approvedBy(doObj.getApprovedBy())
                .createdAt(doObj.getCreatedAt())
                .updatedAt(doObj.getUpdatedAt())
                .lines(lines)
                .build();
    }

    public VoucherDO toDO(Voucher entity) {
        if (entity == null) {
            return null;
        }
        VoucherDO doObj = new VoucherDO();
        doObj.setId(entity.getId());
        doObj.setVoucherNo(entity.getVoucherNo());
        doObj.setType(entity.getType() != null ? entity.getType().name() : null);
        doObj.setVoucherDate(entity.getVoucherDate());
        doObj.setDescription(entity.getDescription());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setApprovedBy(entity.getApprovedBy());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}