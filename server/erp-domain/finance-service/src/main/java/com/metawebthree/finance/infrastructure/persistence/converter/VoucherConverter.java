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
        Voucher entity = new Voucher();
        entity.setId(doObj.getId());
        entity.setVoucherNo(doObj.getVoucherNo());
        entity.setType(Voucher.VoucherType.valueOf(doObj.getType()));
        entity.setVoucherDate(doObj.getVoucherDate());
        entity.setDescription(doObj.getDescription());
        entity.setStatus(Voucher.VoucherStatus.valueOf(doObj.getStatus()));
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setApprovedBy(doObj.getApprovedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        
        if (lineDOs != null && !lineDOs.isEmpty()) {
            List<Voucher.VoucherLine> lines = new ArrayList<>();
            for (VoucherLineDO lineDO : lineDOs) {
                Voucher.VoucherLine line = new Voucher.VoucherLine();
                line.subjectId = lineDO.getSubjectId();
                line.debitAmount = lineDO.getDebitAmount();
                line.creditAmount = lineDO.getCreditAmount();
                lines.add(line);
            }
            entity.setLines(lines);
        }
        
        return entity;
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