package com.metawebthree.traceability.application.query;

import com.metawebthree.traceability.application.dto.ProductInfoDTO;
import com.metawebthree.traceability.application.dto.TraceEventDTO;
import com.metawebthree.traceability.application.dto.TraceRecordDTO;
import com.metawebthree.traceability.domain.entity.ProductInfoDO;
import com.metawebthree.traceability.domain.entity.TraceEventDO;
import com.metawebthree.traceability.domain.entity.TraceRecordDO;
import com.metawebthree.traceability.infrastructure.persistence.mapper.ProductInfoMapper;
import com.metawebthree.traceability.infrastructure.persistence.mapper.TraceEventMapper;
import com.metawebthree.traceability.infrastructure.persistence.mapper.TraceRecordMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class TraceabilityQueryService {

    private final ProductInfoMapper productInfoMapper;
    private final TraceRecordMapper traceRecordMapper;
    private final TraceEventMapper traceEventMapper;

    private enum ProductStatus {
        CREATED,
        PRODUCTION_COMPLETED,
        IN_TRANSIT,
        DELIVERED,
        SOLD
    }

    public TraceRecordDTO getTraceRecord(Long traceId) {
        TraceRecordDO record = traceRecordMapper.selectByTraceId(traceId);
        if (record == null) {
            return null;
        }

        TraceRecordDTO dto = new TraceRecordDTO();
        dto.setTraceId(record.getTraceId());
        dto.setProductId(record.getProductId());
        dto.setProductName(record.getProductName());
        dto.setBatchNumber(record.getBatchNumber());
        dto.setProducer(record.getProducer());
        dto.setProductionTime(record.getProductionTime());
        dto.setStatus(ProductStatus.values()[record.getStatus()].name());

        List<TraceEventDO> events = traceEventMapper.selectByTraceId(traceId);
        List<TraceEventDTO> eventDTOs = events.stream()
            .map(this::toTraceEventDTO)
            .collect(Collectors.toList());
        dto.setEvents(eventDTOs);

        return dto;
    }

    public List<Long> getProductTraceIds(String productId) {
        return traceRecordMapper.selectTraceIdsByProductId(productId);
    }

    public ProductInfoDTO getProductInfo(String productId) {
        ProductInfoDO productInfo = productInfoMapper.selectById(productId);
        if (productInfo == null) {
            return null;
        }

        ProductInfoDTO dto = new ProductInfoDTO();
        dto.setProductId(productInfo.getProductId());
        dto.setProductName(productInfo.getProductName());
        dto.setCategory(productInfo.getCategory());
        dto.setManufacturer(productInfo.getManufacturer());
        dto.setProductionLocation(productInfo.getProductionLocation());
        dto.setProductionDate(productInfo.getProductionDate());
        dto.setIsActive(productInfo.getIsActive());

        return dto;
    }

    public List<TraceEventDTO> getTraceEvents(Long traceId) {
        List<TraceEventDO> events = traceEventMapper.selectByTraceId(traceId);
        return events.stream()
            .map(this::toTraceEventDTO)
            .collect(Collectors.toList());
    }

    public boolean verifyProduct(String productId, String batchNumber) {
        List<TraceRecordDO> records = traceRecordMapper.selectByProductId(productId);
        for (TraceRecordDO record : records) {
            if (record.getBatchNumber() != null && record.getBatchNumber().equals(batchNumber)) {
                return true;
            }
        }
        return false;
    }

    public List<ProductInfoDTO> listProducts(String status, Integer pageNum, Integer pageSize) {
        LambdaQueryWrapper<ProductInfoDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(status != null && !status.isEmpty(), ProductInfoDO::getIsActive, "ACTIVE".equals(status));
        wrapper.orderByDesc(ProductInfoDO::getCreateTime);

        List<ProductInfoDO> products = productInfoMapper.selectList(wrapper);
        int start = (pageNum - 1) * pageSize;
        int end = Math.min(start + pageSize, products.size());
        if (start >= products.size()) {
            return List.of();
        }

        return products.subList(start, end).stream()
            .map(this::toProductInfoDTO)
            .collect(Collectors.toList());
    }

    public long countProducts(String status) {
        LambdaQueryWrapper<ProductInfoDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(status != null && !status.isEmpty(), ProductInfoDO::getIsActive, "ACTIVE".equals(status));
        return productInfoMapper.selectCount(wrapper);
    }

    private ProductInfoDTO toProductInfoDTO(ProductInfoDO doObj) {
        ProductInfoDTO dto = new ProductInfoDTO();
        dto.setProductId(doObj.getProductId());
        dto.setProductName(doObj.getProductName());
        dto.setCategory(doObj.getCategory());
        dto.setManufacturer(doObj.getManufacturer());
        dto.setProductionLocation(doObj.getProductionLocation());
        dto.setProductionDate(doObj.getProductionDate());
        dto.setIsActive(doObj.getIsActive());
        return dto;
    }

    private TraceEventDTO toTraceEventDTO(TraceEventDO event) {
        TraceEventDTO dto = new TraceEventDTO();
        dto.setTraceId(event.getTraceId());
        dto.setEventType(event.getEventType());
        dto.setDescription(event.getDescription());
        dto.setLocation(event.getLocation());
        dto.setOperator(event.getOperator());
        dto.setTimestamp(event.getTimestamp());
        dto.setExtraData(event.getExtraData());
        return dto;
    }
}