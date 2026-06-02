package com.metawebthree.finance.adapter.http;

import com.metawebthree.finance.application.command.arap.ArApCommandService;
import com.metawebthree.finance.application.command.arap.dto.ArCreateCommand;
import com.metawebthree.finance.application.command.arap.dto.ArReceiveCommand;
import com.metawebthree.finance.application.command.arap.dto.ApCreateCommand;
import com.metawebthree.finance.application.command.arap.dto.ApPayCommand;
import com.metawebthree.finance.application.query.arap.ArApQueryService;
import com.metawebthree.finance.application.query.arap.dto.ArQueryResult;
import com.metawebthree.finance.application.query.arap.dto.ApQueryResult;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/finance/arap")
public class ArApController {
    private final ArApCommandService commandService;
    private final ArApQueryService queryService;

    public ArApController(ArApCommandService commandService, ArApQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping("/ar")
    public ResponseEntity<ArQueryResult> createAr(@RequestBody ArCreateCommand command) {
        var result = commandService.createAr(command);
        return ResponseEntity.ok(ArQueryResult.fromEntity(result));
    }

    @PostMapping("/ar/receive")
    public ResponseEntity<ArQueryResult> receiveAr(@RequestBody ArReceiveCommand command) {
        var result = commandService.receiveAr(command);
        return ResponseEntity.ok(ArQueryResult.fromEntity(result));
    }

    @PostMapping("/ar/{id}/writeoff")
    public ResponseEntity<ArQueryResult> writeOffAr(
            @PathVariable Long id,
            @RequestParam java.math.BigDecimal amount) {
        var result = commandService.writeOffAr(id, amount);
        return ResponseEntity.ok(ArQueryResult.fromEntity(result));
    }

    @GetMapping("/ar/{id}")
    public ResponseEntity<ArQueryResult> getArById(@PathVariable Long id) {
        var result = queryService.getArById(id);
        return result != null ? ResponseEntity.ok(result) : ResponseEntity.notFound().build();
    }

    @GetMapping("/ar/code/{code}")
    public ResponseEntity<ArQueryResult> getArByCode(@PathVariable String code) {
        var result = queryService.getArByCode(code);
        return result != null ? ResponseEntity.ok(result) : ResponseEntity.notFound().build();
    }

    @GetMapping("/ar/customer/{customerId}")
    public ResponseEntity<List<ArQueryResult>> getArByCustomerId(@PathVariable Long customerId) {
        return ResponseEntity.ok(queryService.getArByCustomerId(customerId));
    }

    @GetMapping("/ar/status/{status}")
    public ResponseEntity<List<ArQueryResult>> getArByStatus(@PathVariable String status) {
        return ResponseEntity.ok(queryService.getArByStatus(status));
    }

    @GetMapping("/ar/overdue")
    public ResponseEntity<List<ArQueryResult>> getOverdueAr() {
        return ResponseEntity.ok(queryService.getOverdueAr());
    }

    @GetMapping("/ar/list")
    public ResponseEntity<List<ArQueryResult>> listAllAr() {
        return ResponseEntity.ok(queryService.getAllAr());
    }

    @PostMapping("/ap")
    public ResponseEntity<ApQueryResult> createAp(@RequestBody ApCreateCommand command) {
        var result = commandService.createAp(command);
        return ResponseEntity.ok(ApQueryResult.fromEntity(result));
    }

    @PostMapping("/ap/pay")
    public ResponseEntity<ApQueryResult> payAp(@RequestBody ApPayCommand command) {
        var result = commandService.payAp(command);
        return ResponseEntity.ok(ApQueryResult.fromEntity(result));
    }

    @PostMapping("/ap/{id}/writeoff")
    public ResponseEntity<ApQueryResult> writeOffAp(
            @PathVariable Long id,
            @RequestParam java.math.BigDecimal amount) {
        var result = commandService.writeOffAp(id, amount);
        return ResponseEntity.ok(ApQueryResult.fromEntity(result));
    }

    @GetMapping("/ap/{id}")
    public ResponseEntity<ApQueryResult> getApById(@PathVariable Long id) {
        var result = queryService.getApById(id);
        return result != null ? ResponseEntity.ok(result) : ResponseEntity.notFound().build();
    }

    @GetMapping("/ap/code/{code}")
    public ResponseEntity<ApQueryResult> getApByCode(@PathVariable String code) {
        var result = queryService.getApByCode(code);
        return result != null ? ResponseEntity.ok(result) : ResponseEntity.notFound().build();
    }

    @GetMapping("/ap/supplier/{supplierId}")
    public ResponseEntity<List<ApQueryResult>> getApBySupplierId(@PathVariable Long supplierId) {
        return ResponseEntity.ok(queryService.getApBySupplierId(supplierId));
    }

    @GetMapping("/ap/status/{status}")
    public ResponseEntity<List<ApQueryResult>> getApByStatus(@PathVariable String status) {
        return ResponseEntity.ok(queryService.getApByStatus(status));
    }

    @GetMapping("/ap/overdue")
    public ResponseEntity<List<ApQueryResult>> getOverdueAp() {
        return ResponseEntity.ok(queryService.getOverdueAp());
    }

    @GetMapping("/ap/list")
    public ResponseEntity<List<ApQueryResult>> listAllAp() {
        return ResponseEntity.ok(queryService.getAllAp());
    }

    @PostMapping("/check-overdue")
    public ResponseEntity<Map<String, String>> checkOverdue() {
        commandService.checkOverdueAr();
        commandService.checkOverdueAp();
        return ResponseEntity.ok(Map.of("status", "completed"));
    }
}