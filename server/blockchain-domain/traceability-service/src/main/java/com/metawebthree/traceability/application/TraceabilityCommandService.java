package com.metawebthree.traceability.application;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.traceability.application.command.AddTraceEventCommand;
import com.metawebthree.traceability.application.command.CreateTraceCommand;
import com.metawebthree.traceability.application.command.RegisterProductCommand;
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
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class TraceabilityCommandService {

    private final ProductInfoMapper productInfoMapper;
    private final TraceRecordMapper traceRecordMapper;
    private final TraceEventMapper traceEventMapper;

    @Transactional
    public void registerProduct(RegisterProductCommand command) {
        ProductInfoDO productInfo = new ProductInfoDO();
        productInfo.setProductId(command.getProductId());
        productInfo.setProductName(command.getProductName());
        productInfo.setCategory(command.getCategory());
        productInfo.setManufacturer(command.getManufacturer());
        productInfo.setProductionLocation(command.getProductionLocation());
        productInfo.setProductionDate(command.getProductionDate());
        productInfo.setIsActive(true);
        productInfo.setCreateTime(LocalDateTime.now());
        productInfo.setUpdateTime(LocalDateTime.now());
        productInfoMapper.insert(productInfo);
    }

    @Transactional
    public Long createTraceRecord(CreateTraceCommand command) {
        ProductInfoDO productInfo = productInfoMapper.selectById(command.getProductId());
        if (productInfo == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Product not registered");
        }

        Long maxTraceId = traceRecordMapper.selectMaxTraceId();
        long newTraceId = (maxTraceId == null) ? 1 : maxTraceId + 1;

        TraceRecordDO record = new TraceRecordDO();
        record.setTraceId(newTraceId);
        record.setProductId(command.getProductId());
        record.setProductName(productInfo.getProductName());
        record.setBatchNumber(command.getBatchNumber());
        record.setProducer("SYSTEM");
        record.setProductionTime(LocalDateTime.now());
        record.setStatus(0);
        record.setCreateTime(LocalDateTime.now());
        record.setUpdateTime(LocalDateTime.now());
        traceRecordMapper.insert(record);

        return newTraceId;
    }

    @Transactional
    public void addTraceEvent(AddTraceEventCommand command) {
        TraceRecordDO record = traceRecordMapper.selectByTraceId(command.getTraceId());
        if (record == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Trace record not found");
        }

        TraceEventDO event = new TraceEventDO();
        event.setTraceId(command.getTraceId());
        event.setEventType(command.getEventType());
        event.setDescription(command.getDescription());
        event.setLocation(command.getLocation());
        event.setOperator("SYSTEM");
        event.setTimestamp(LocalDateTime.now());
        event.setExtraData(command.getExtraData());
        event.setCreateTime(LocalDateTime.now());
        traceEventMapper.insert(event);
    }

    @Transactional
    public void recordProduction(Long traceId, String location, String qualityInfo) {
        addTraceEvent(new AddTraceEventCommand() {{
            setTraceId(traceId);
            setEventType("PRODUCTION_COMPLETED");
            setDescription(qualityInfo);
            setLocation(location);
            setExtraData("");
        }});

        TraceRecordDO record = traceRecordMapper.selectByTraceId(traceId);
        record.setStatus(1);
        record.setUpdateTime(LocalDateTime.now());
        traceRecordMapper.updateById(record);
    }

    @Transactional
    public void recordTransportation(Long traceId, String fromLocation, String toLocation, String carrierInfo) {
        String description = "From: " + fromLocation + " To: " + toLocation + " Carrier: " + carrierInfo;

        addTraceEvent(new AddTraceEventCommand() {{
            setTraceId(traceId);
            setEventType("TRANSPORTATION");
            setDescription(description);
            setLocation(toLocation);
            setExtraData(carrierInfo);
        }});

        TraceRecordDO record = traceRecordMapper.selectByTraceId(traceId);
        record.setStatus(2);
        record.setUpdateTime(LocalDateTime.now());
        traceRecordMapper.updateById(record);
    }

    @Transactional
    public void recordDelivery(Long traceId, String location, String receiverInfo) {
        addTraceEvent(new AddTraceEventCommand() {{
            setTraceId(traceId);
            setEventType("DELIVERED");
            setDescription(receiverInfo);
            setLocation(location);
            setExtraData("");
        }});

        TraceRecordDO record = traceRecordMapper.selectByTraceId(traceId);
        record.setStatus(3);
        record.setUpdateTime(LocalDateTime.now());
        traceRecordMapper.updateById(record);
    }

    @Transactional
    public void recordSale(Long traceId, String buyerAddress, String saleLocation, Long price) {
        String description = "Sold to: " + buyerAddress + " Price: " + price;

        addTraceEvent(new AddTraceEventCommand() {{
            setTraceId(traceId);
            setEventType("SOLD");
            setDescription(description);
            setLocation(saleLocation);
            setExtraData(String.valueOf(price));
        }});

        TraceRecordDO record = traceRecordMapper.selectByTraceId(traceId);
        record.setStatus(4);
        record.setUpdateTime(LocalDateTime.now());
        traceRecordMapper.updateById(record);
    }
}