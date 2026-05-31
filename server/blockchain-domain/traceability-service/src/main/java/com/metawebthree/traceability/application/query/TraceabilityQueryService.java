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

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class TraceabilityQueryService {

    private final ProductInfoMapper productInfoMapper;
    private final TraceRecordMapper traceRecordMapper;
    private final TraceEventMapper traceEventMapper;

    private static final String[] STATUS_NAMES = {
        "Created", "ProductionCompleted", "InTransit", "Delivered", "Sold"
    };

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
        dto.setStatus(STATUS_NAMES[record.getStatus()]);

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
            if (record.getBatchNumber().equals(batchNumber)) {
                return true;
            }
        }
        return false;
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