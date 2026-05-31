package com.metawebthree.traceability.interfaces.controller;

import com.metawebthree.traceability.application.TraceabilityCommandService;
import com.metawebthree.traceability.application.command.AddTraceEventCommand;
import com.metawebthree.traceability.application.command.CreateTraceCommand;
import com.metawebthree.traceability.application.command.RegisterProductCommand;
import com.metawebthree.traceability.application.dto.ProductInfoDTO;
import com.metawebthree.traceability.application.dto.TraceEventDTO;
import com.metawebthree.traceability.application.dto.TraceRecordDTO;
import com.metawebthree.traceability.application.query.TraceabilityQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/traceability")
@RequiredArgsConstructor
public class TraceabilityController {

    private final TraceabilityCommandService commandService;
    private final TraceabilityQueryService queryService;

    @PostMapping("/product")
    public ResponseEntity<Void> registerProduct(@RequestBody RegisterProductCommand command) {
        commandService.registerProduct(command);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/trace")
    public ResponseEntity<Map<String, Long>> createTraceRecord(@RequestBody CreateTraceCommand command) {
        Long traceId = commandService.createTraceRecord(command);
        return ResponseEntity.ok(Map.of("traceId", traceId));
    }

    @PostMapping("/trace/{traceId}/event")
    public ResponseEntity<Void> addTraceEvent(
            @PathVariable Long traceId,
            @RequestBody AddTraceEventCommand command) {
        command.setTraceId(traceId);
        commandService.addTraceEvent(command);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/trace/{traceId}/production")
    public ResponseEntity<Void> recordProduction(
            @PathVariable Long traceId,
            @RequestBody Map<String, String> request) {
        commandService.recordProduction(
            traceId,
            request.get("location"),
            request.get("qualityInfo")
        );
        return ResponseEntity.ok().build();
    }

    @PostMapping("/trace/{traceId}/transportation")
    public ResponseEntity<Void> recordTransportation(
            @PathVariable Long traceId,
            @RequestBody Map<String, String> request) {
        commandService.recordTransportation(
            traceId,
            request.get("fromLocation"),
            request.get("toLocation"),
            request.get("carrierInfo")
        );
        return ResponseEntity.ok().build();
    }

    @PostMapping("/trace/{traceId}/delivery")
    public ResponseEntity<Void> recordDelivery(
            @PathVariable Long traceId,
            @RequestBody Map<String, String> request) {
        commandService.recordDelivery(
            traceId,
            request.get("location"),
            request.get("receiverInfo")
        );
        return ResponseEntity.ok().build();
    }

    @PostMapping("/trace/{traceId}/sale")
    public ResponseEntity<Void> recordSale(
            @PathVariable Long traceId,
            @RequestBody Map<String, Object> request) {
        commandService.recordSale(
            traceId,
            (String) request.get("buyerAddress"),
            (String) request.get("saleLocation"),
            ((Number) request.get("price")).longValue()
        );
        return ResponseEntity.ok().build();
    }

    @GetMapping("/trace/{traceId}")
    public ResponseEntity<TraceRecordDTO> getTraceRecord(@PathVariable Long traceId) {
        TraceRecordDTO dto = queryService.getTraceRecord(traceId);
        if (dto == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(dto);
    }

    @GetMapping("/product/{productId}/traces")
    public ResponseEntity<List<Long>> getProductTraceIds(@PathVariable String productId) {
        List<Long> traceIds = queryService.getProductTraceIds(productId);
        return ResponseEntity.ok(traceIds);
    }

    @GetMapping("/product/{productId}")
    public ResponseEntity<ProductInfoDTO> getProductInfo(@PathVariable String productId) {
        ProductInfoDTO dto = queryService.getProductInfo(productId);
        if (dto == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(dto);
    }

    @GetMapping("/trace/{traceId}/events")
    public ResponseEntity<List<TraceEventDTO>> getTraceEvents(@PathVariable Long traceId) {
        List<TraceEventDTO> events = queryService.getTraceEvents(traceId);
        return ResponseEntity.ok(events);
    }

    @GetMapping("/verify")
    public ResponseEntity<Map<String, Boolean>> verifyProduct(
            @RequestParam String productId,
            @RequestParam String batchNumber) {
        boolean verified = queryService.verifyProduct(productId, batchNumber);
        return ResponseEntity.ok(Map.of("verified", verified));
    }
}