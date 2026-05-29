package com.metawebthree.invoice.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.invoice.domain.entity.Invoice;
import com.metawebthree.invoice.domain.repository.InvoiceRepository;
import com.metawebthree.invoice.infrastructure.persistence.converter.InvoiceConverter;
import com.metawebthree.invoice.infrastructure.persistence.dataobject.InvoiceDO;
import com.metawebthree.invoice.infrastructure.persistence.mapper.InvoiceMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class InvoiceRepositoryImpl implements InvoiceRepository {

    private final InvoiceMapper invoiceMapper;
    private final InvoiceConverter invoiceConverter;

    public InvoiceRepositoryImpl(InvoiceMapper invoiceMapper, InvoiceConverter invoiceConverter) {
        this.invoiceMapper = invoiceMapper;
        this.invoiceConverter = invoiceConverter;
    }

    @Override
    public Optional<Invoice> findById(Long id) {
        InvoiceDO invoiceDO = invoiceMapper.selectById(id);
        return Optional.ofNullable(invoiceConverter.toEntity(invoiceDO));
    }

    @Override
    public Optional<Invoice> findByInvoiceNo(String invoiceNo) {
        LambdaQueryWrapper<InvoiceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InvoiceDO::getInvoiceNo, invoiceNo);
        InvoiceDO invoiceDO = invoiceMapper.selectOne(wrapper);
        return Optional.ofNullable(invoiceConverter.toEntity(invoiceDO));
    }

    @Override
    public List<Invoice> findByStatus(Invoice.InvoiceStatus status) {
        LambdaQueryWrapper<InvoiceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InvoiceDO::getStatus, status.name());
        List<InvoiceDO> invoiceDOs = invoiceMapper.selectList(wrapper);
        return invoiceDOs.stream()
                .map(invoiceConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Invoice> findByCustomerId(Long customerId) {
        LambdaQueryWrapper<InvoiceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InvoiceDO::getCustomerId, customerId);
        List<InvoiceDO> invoiceDOs = invoiceMapper.selectList(wrapper);
        return invoiceDOs.stream()
                .map(invoiceConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Invoice> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end) {
        LambdaQueryWrapper<InvoiceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(InvoiceDO::getIssueDate, start, end);
        List<InvoiceDO> invoiceDOs = invoiceMapper.selectList(wrapper);
        return invoiceDOs.stream()
                .map(invoiceConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<Invoice> findAll() {
        List<InvoiceDO> invoiceDOs = invoiceMapper.selectList(null);
        return invoiceDOs.stream()
                .map(invoiceConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public void save(Invoice invoice) {
        InvoiceDO invoiceDO = invoiceConverter.toDO(invoice);
        if (invoice.getId() == null) {
            invoiceMapper.insert(invoiceDO);
            invoice.setId(invoiceDO.getId());
        } else {
            invoiceMapper.updateById(invoiceDO);
        }
    }

    @Override
    public void update(Invoice invoice) {
        InvoiceDO invoiceDO = invoiceConverter.toDO(invoice);
        invoiceMapper.updateById(invoiceDO);
    }

    @Override
    public void delete(Long id) {
        invoiceMapper.deleteById(id);
    }
}