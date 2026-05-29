package com.metawebthree.finance.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.finance.domain.entity.Voucher;
import com.metawebthree.finance.domain.repository.VoucherRepository;
import com.metawebthree.finance.infrastructure.persistence.converter.VoucherConverter;
import com.metawebthree.finance.infrastructure.persistence.dataobject.VoucherDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.VoucherLineDO;
import com.metawebthree.finance.infrastructure.persistence.mapper.VoucherLineMapper;
import com.metawebthree.finance.infrastructure.persistence.mapper.VoucherMapper;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class VoucherRepositoryImpl implements VoucherRepository {

    private final VoucherMapper voucherMapper;
    private final VoucherLineMapper voucherLineMapper;
    private final VoucherConverter voucherConverter;

    public VoucherRepositoryImpl(VoucherMapper voucherMapper, 
                                  VoucherLineMapper voucherLineMapper,
                                  VoucherConverter voucherConverter) {
        this.voucherMapper = voucherMapper;
        this.voucherLineMapper = voucherLineMapper;
        this.voucherConverter = voucherConverter;
    }

    @Override
    public Optional<Voucher> findById(Long id) {
        VoucherDO voucherDO = voucherMapper.selectById(id);
        if (voucherDO == null) {
            return Optional.empty();
        }
        List<VoucherLineDO> lineDOs = voucherLineMapper.selectByVoucherId(id);
        return Optional.ofNullable(voucherConverter.toEntity(voucherDO, lineDOs));
    }

    @Override
    public Optional<Voucher> findByVoucherNo(String voucherNo) {
        LambdaQueryWrapper<VoucherDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(VoucherDO::getVoucherNo, voucherNo);
        VoucherDO voucherDO = voucherMapper.selectOne(wrapper);
        if (voucherDO == null) {
            return Optional.empty();
        }
        List<VoucherLineDO> lineDOs = voucherLineMapper.selectByVoucherId(voucherDO.getId());
        return Optional.ofNullable(voucherConverter.toEntity(voucherDO, lineDOs));
    }

    @Override
    public List<Voucher> findByStatus(Voucher.VoucherStatus status) {
        LambdaQueryWrapper<VoucherDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(VoucherDO::getStatus, status.name());
        List<VoucherDO> voucherDOs = voucherMapper.selectList(wrapper);
        return voucherDOs.stream()
                .map(vo -> {
                    List<VoucherLineDO> lineDOs = voucherLineMapper.selectByVoucherId(vo.getId());
                    return voucherConverter.toEntity(vo, lineDOs);
                })
                .collect(Collectors.toList());
    }

    @Override
    public List<Voucher> findByVoucherDateBetween(java.time.LocalDateTime start, java.time.LocalDateTime end) {
        LambdaQueryWrapper<VoucherDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(VoucherDO::getVoucherDate, start, end);
        List<VoucherDO> voucherDOs = voucherMapper.selectList(wrapper);
        return voucherDOs.stream()
                .map(vo -> {
                    List<VoucherLineDO> lineDOs = voucherLineMapper.selectByVoucherId(vo.getId());
                    return voucherConverter.toEntity(vo, lineDOs);
                })
                .collect(Collectors.toList());
    }

    @Override
    public List<Voucher> findByCreatedBy(String createdBy) {
        LambdaQueryWrapper<VoucherDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(VoucherDO::getCreatedBy, createdBy);
        List<VoucherDO> voucherDOs = voucherMapper.selectList(wrapper);
        return voucherDOs.stream()
                .map(vo -> {
                    List<VoucherLineDO> lineDOs = voucherLineMapper.selectByVoucherId(vo.getId());
                    return voucherConverter.toEntity(vo, lineDOs);
                })
                .collect(Collectors.toList());
    }

    @Override
    public List<Voucher> findByType(Voucher.VoucherType type) {
        LambdaQueryWrapper<VoucherDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(VoucherDO::getType, type.name());
        List<VoucherDO> voucherDOs = voucherMapper.selectList(wrapper);
        return voucherDOs.stream()
                .map(vo -> {
                    List<VoucherLineDO> lineDOs = voucherLineMapper.selectByVoucherId(vo.getId());
                    return voucherConverter.toEntity(vo, lineDOs);
                })
                .collect(Collectors.toList());
    }

    @Override
    public List<Voucher> findAll() {
        List<VoucherDO> voucherDOs = voucherMapper.selectList(null);
        return voucherDOs.stream()
                .map(vo -> {
                    List<VoucherLineDO> lineDOs = voucherLineMapper.selectByVoucherId(vo.getId());
                    return voucherConverter.toEntity(vo, lineDOs);
                })
                .collect(Collectors.toList());
    }

    @Override
    @Transactional
    public void save(Voucher voucher) {
        VoucherDO voucherDO = voucherConverter.toDO(voucher);
        if (voucher.getId() == null) {
            voucherMapper.insert(voucherDO);
            voucher.setId(voucherDO.getId());
        } else {
            voucherMapper.updateById(voucherDO);
        }
        
        // Save lines
        if (voucher.getLines() != null && !voucher.getLines().isEmpty()) {
            // Delete existing lines if updating
            if (voucher.getId() != null) {
                LambdaQueryWrapper<VoucherLineDO> wrapper = new LambdaQueryWrapper<>();
                wrapper.eq(VoucherLineDO::getVoucherId, voucher.getId());
                voucherLineMapper.delete(wrapper);
            }
            
            // Insert new lines
            for (Voucher.VoucherLine line : voucher.getLines()) {
                VoucherLineDO lineDO = new VoucherLineDO();
                lineDO.setVoucherId(voucher.getId());
                lineDO.setSubjectId(line.subjectId);
                lineDO.setDebitAmount(line.debitAmount);
                lineDO.setCreditAmount(line.creditAmount);
                voucherLineMapper.insert(lineDO);
            }
        }
    }

    @Override
    @Transactional
    public void update(Voucher voucher) {
        VoucherDO voucherDO = voucherConverter.toDO(voucher);
        voucherMapper.updateById(voucherDO);
        
        // Update lines - delete and re-insert
        if (voucher.getLines() != null) {
            LambdaQueryWrapper<VoucherLineDO> wrapper = new LambdaQueryWrapper<>();
            wrapper.eq(VoucherLineDO::getVoucherId, voucher.getId());
            voucherLineMapper.delete(wrapper);
            
            for (Voucher.VoucherLine line : voucher.getLines()) {
                VoucherLineDO lineDO = new VoucherLineDO();
                lineDO.setVoucherId(voucher.getId());
                lineDO.setSubjectId(line.subjectId);
                lineDO.setDebitAmount(line.debitAmount);
                lineDO.setCreditAmount(line.creditAmount);
                voucherLineMapper.insert(lineDO);
            }
        }
    }

    @Override
    @Transactional
    public void delete(Long id) {
        // Delete lines first
        LambdaQueryWrapper<VoucherLineDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(VoucherLineDO::getVoucherId, id);
        voucherLineMapper.delete(wrapper);
        
        // Delete voucher
        voucherMapper.deleteById(id);
    }
}