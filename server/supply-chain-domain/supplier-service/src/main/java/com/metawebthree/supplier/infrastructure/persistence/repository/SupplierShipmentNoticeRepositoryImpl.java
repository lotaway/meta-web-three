package com.metawebthree.supplier.infrastructure.persistence.repository;

import com.metawebthree.supplier.domain.entity.SupplierShipmentNotice;
import com.metawebthree.supplier.domain.entity.SupplierShipmentNoticeItem;
import com.metawebthree.supplier.domain.repository.SupplierShipmentNoticeRepository;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierShipmentNoticeDO;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierShipmentNoticeItemDO;
import com.metawebthree.supplier.infrastructure.persistence.mapper.SupplierShipmentNoticeItemMapper;
import com.metawebthree.supplier.infrastructure.persistence.mapper.SupplierShipmentNoticeMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Repository
public class SupplierShipmentNoticeRepositoryImpl implements SupplierShipmentNoticeRepository {

    private final SupplierShipmentNoticeMapper shipmentNoticeMapper;
    private final SupplierShipmentNoticeItemMapper shipmentNoticeItemMapper;

    public SupplierShipmentNoticeRepositoryImpl(
            SupplierShipmentNoticeMapper shipmentNoticeMapper,
            SupplierShipmentNoticeItemMapper shipmentNoticeItemMapper) {
        this.shipmentNoticeMapper = shipmentNoticeMapper;
        this.shipmentNoticeItemMapper = shipmentNoticeItemMapper;
    }

    @Override
    public SupplierShipmentNotice save(SupplierShipmentNotice notice) {
        SupplierShipmentNoticeDO noticeDO = toDO(notice);
        if (notice.getId() == null) {
            noticeDO.setCreatedAt(LocalDateTime.now());
            shipmentNoticeMapper.insert(noticeDO);
            notice.setId(noticeDO.getId());
        } else {
            noticeDO.setUpdatedAt(LocalDateTime.now());
            shipmentNoticeMapper.updateById(noticeDO);
        }
        
        // 保存明细
        if (notice.getItems() != null && !notice.getItems().isEmpty()) {
            for (SupplierShipmentNoticeItem item : notice.getItems()) {
                SupplierShipmentNoticeItemDO itemDO = toItemDO(item);
                itemDO.setNoticeId(notice.getId());
                if (item.getId() == null) {
                    itemDO.setCreatedAt(LocalDateTime.now());
                    shipmentNoticeItemMapper.insert(itemDO);
                    item.setId(itemDO.getId());
                } else {
                    itemDO.setUpdatedAt(LocalDateTime.now());
                    shipmentNoticeItemMapper.updateById(itemDO);
                }
            }
        }
        
        return notice;
    }

    @Override
    public Optional<SupplierShipmentNotice> findById(Long id) {
        SupplierShipmentNoticeDO noticeDO = shipmentNoticeMapper.selectById(id);
        if (noticeDO == null) {
            return Optional.empty();
        }
        SupplierShipmentNotice notice = toEntity(noticeDO);
        // 加载明细
        List<SupplierShipmentNoticeItemDO> itemDOs = shipmentNoticeItemMapper.selectByNoticeId(id);
        notice.setItems(itemDOs.stream().map(this::toItemEntity).toList());
        return Optional.of(notice);
    }

    @Override
    public Optional<SupplierShipmentNotice> findByNoticeNo(String noticeNo) {
        SupplierShipmentNoticeDO noticeDO = shipmentNoticeMapper.selectByNoticeNo(noticeNo);
        if (noticeDO == null) {
            return Optional.empty();
        }
        SupplierShipmentNotice notice = toEntity(noticeDO);
        List<SupplierShipmentNoticeItemDO> itemDOs = shipmentNoticeItemMapper.selectByNoticeId(noticeDO.getId());
        notice.setItems(itemDOs.stream().map(this::toItemEntity).toList());
        return Optional.of(notice);
    }

    @Override
    public List<SupplierShipmentNotice> findBySupplierCode(String supplierCode) {
        List<SupplierShipmentNoticeDO> list = shipmentNoticeMapper.selectBySupplierCode(supplierCode);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<SupplierShipmentNotice> findByOrderNo(String orderNo) {
        List<SupplierShipmentNoticeDO> list = shipmentNoticeMapper.selectByOrderNo(orderNo);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<SupplierShipmentNotice> findBySupplierCodeAndStatus(String supplierCode, String status) {
        List<SupplierShipmentNoticeDO> list = shipmentNoticeMapper.selectBySupplierCodeAndStatus(supplierCode, status);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<SupplierShipmentNotice> findBySupplierCodeAndStatusIn(String supplierCode, List<String> statuses) {
        // 简化实现，查询所有该供应商的记录然后过滤
        List<SupplierShipmentNoticeDO> list = shipmentNoticeMapper.selectBySupplierCode(supplierCode);
        return list.stream()
                .filter(doItem -> statuses.contains(doItem.getStatus()))
                .map(this::toEntity)
                .toList();
    }

    @Override
    public void deleteById(Long id) {
        shipmentNoticeItemMapper.deleteById(id);
        shipmentNoticeMapper.deleteById(id);
    }

    private SupplierShipmentNoticeDO toDO(SupplierShipmentNotice notice) {
        SupplierShipmentNoticeDO doItem = new SupplierShipmentNoticeDO();
        doItem.setId(notice.getId());
        doItem.setNoticeNo(notice.getNoticeNo());
        doItem.setSupplierCode(notice.getSupplierCode());
        doItem.setOrderNo(notice.getOrderNo());
        doItem.setWarehouseId(notice.getWarehouseId());
        doItem.setExpectedShipmentDate(notice.getExpectedShipmentDate());
        doItem.setActualShipmentDate(notice.getActualShipmentDate());
        doItem.setShipmentMethod(notice.getShipmentMethod());
        doItem.setCarrierName(notice.getCarrierName());
        doItem.setCarrierContact(notice.getCarrierContact());
        doItem.setTrackingNumber(notice.getTrackingNumber());
        doItem.setVehicleNumber(notice.getVehicleNumber());
        doItem.setDriverName(notice.getDriverName());
        doItem.setDriverPhone(notice.getDriverPhone());
        doItem.setTotalQuantity(notice.getTotalQuantity());
        doItem.setTotalWeight(notice.getTotalWeight());
        doItem.setTotalVolume(notice.getTotalVolume());
        doItem.setStatus(notice.getStatus());
        doItem.setRemark(notice.getRemark());
        doItem.setConfirmer(notice.getConfirmer());
        doItem.setConfirmedAt(notice.getConfirmedAt());
        doItem.setCreatedAt(notice.getCreatedAt());
        doItem.setUpdatedAt(notice.getUpdatedAt());
        return doItem;
    }

    private SupplierShipmentNotice toEntity(SupplierShipmentNoticeDO doItem) {
        if (doItem == null) {
            return null;
        }
        SupplierShipmentNotice notice = new SupplierShipmentNotice();
        notice.setId(doItem.getId());
        notice.setNoticeNo(doItem.getNoticeNo());
        notice.setSupplierCode(doItem.getSupplierCode());
        notice.setOrderNo(doItem.getOrderNo());
        notice.setWarehouseId(doItem.getWarehouseId());
        notice.setExpectedShipmentDate(doItem.getExpectedShipmentDate());
        notice.setActualShipmentDate(doItem.getActualShipmentDate());
        notice.setShipmentMethod(doItem.getShipmentMethod());
        notice.setCarrierName(doItem.getCarrierName());
        notice.setCarrierContact(doItem.getCarrierContact());
        notice.setTrackingNumber(doItem.getTrackingNumber());
        notice.setVehicleNumber(doItem.getVehicleNumber());
        notice.setDriverName(doItem.getDriverName());
        notice.setDriverPhone(doItem.getDriverPhone());
        notice.setTotalQuantity(doItem.getTotalQuantity());
        notice.setTotalWeight(doItem.getTotalWeight());
        notice.setTotalVolume(doItem.getTotalVolume());
        notice.setStatus(doItem.getStatus());
        notice.setRemark(doItem.getRemark());
        notice.setConfirmer(doItem.getConfirmer());
        notice.setConfirmedAt(doItem.getConfirmedAt());
        notice.setCreatedAt(doItem.getCreatedAt());
        notice.setUpdatedAt(doItem.getUpdatedAt());
        return notice;
    }

    private SupplierShipmentNoticeItemDO toItemDO(SupplierShipmentNoticeItem item) {
        SupplierShipmentNoticeItemDO doItem = new SupplierShipmentNoticeItemDO();
        doItem.setId(item.getId());
        doItem.setNoticeId(item.getNoticeId());
        doItem.setProductCode(item.getProductCode());
        doItem.setProductName(item.getProductName());
        doItem.setUnit(item.getUnit());
        doItem.setQuantity(item.getQuantity());
        doItem.setWeight(item.getWeight());
        doItem.setVolume(item.getVolume());
        doItem.setBatchNo(item.getBatchNo());
        doItem.setProductionDate(item.getProductionDate());
        doItem.setExpiryDate(item.getExpiryDate());
        doItem.setCreatedAt(item.getCreatedAt());
        doItem.setUpdatedAt(item.getUpdatedAt());
        return doItem;
    }

    private SupplierShipmentNoticeItem toItemEntity(SupplierShipmentNoticeItemDO doItem) {
        if (doItem == null) {
            return null;
        }
        SupplierShipmentNoticeItem item = new SupplierShipmentNoticeItem();
        item.setId(doItem.getId());
        item.setNoticeId(doItem.getNoticeId());
        item.setProductCode(doItem.getProductCode());
        item.setProductName(doItem.getProductName());
        item.setUnit(doItem.getUnit());
        item.setQuantity(doItem.getQuantity());
        item.setWeight(doItem.getWeight());
        item.setVolume(doItem.getVolume());
        item.setBatchNo(doItem.getBatchNo());
        item.setProductionDate(doItem.getProductionDate());
        item.setExpiryDate(doItem.getExpiryDate());
        item.setCreatedAt(doItem.getCreatedAt());
        item.setUpdatedAt(doItem.getUpdatedAt());
        return item;
    }
}