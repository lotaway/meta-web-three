package com.metawebthree.finance.adapter.http;

import com.metawebthree.finance.application.command.ledger.LedgerCommandService;
import com.metawebthree.finance.application.command.ledger.dto.LedgerGenerateCommand;
import com.metawebthree.finance.application.query.ledger.LedgerQueryService;
import com.metawebthree.finance.domain.entity.ledger.GeneralLedger;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/finance/ledger")
public class LedgerController {
    private final LedgerCommandService commandService;
    private final LedgerQueryService queryService;

    public LedgerController(LedgerCommandService commandService, LedgerQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping("/generate")
    public ResponseEntity<GeneralLedger> generateFromVoucher(@RequestBody LedgerGenerateCommand command) {
        var result = commandService.generateLedgerFromVoucher(command);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/{id}/post")
    public ResponseEntity<GeneralLedger> postLedger(@PathVariable Long id) {
        var result = commandService.postLedger(id);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/{id}/close")
    public ResponseEntity<GeneralLedger> closeLedger(@PathVariable Long id) {
        var result = commandService.closeLedger(id);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/{id}")
    public ResponseEntity<GeneralLedger> getLedgerById(@PathVariable Long id) {
        return queryService.getLedgerById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/no/{ledgerNo}")
    public ResponseEntity<GeneralLedger> getLedgerByNo(@PathVariable String ledgerNo) {
        return queryService.getLedgerByNo(ledgerNo)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/period")
    public ResponseEntity<GeneralLedger> getLedgerByPeriod(
            @RequestParam Integer year,
            @RequestParam Integer month) {
        return queryService.getLedgerByPeriod(year, month)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/status/{status}")
    public ResponseEntity<List<GeneralLedger>> getLedgersByStatus(@PathVariable String status) {
        var ledgerStatus = GeneralLedger.LedgerStatus.valueOf(status.toUpperCase());
        return ResponseEntity.ok(queryService.getLedgersByStatus(ledgerStatus));
    }

    @GetMapping("/period-range")
    public ResponseEntity<List<GeneralLedger>> getLedgersByPeriodBetween(
            @RequestParam Integer startYear,
            @RequestParam Integer startMonth,
            @RequestParam Integer endYear,
            @RequestParam Integer endMonth) {
        return ResponseEntity.ok(queryService.getLedgersByPeriodBetween(
                startYear, startMonth, endYear, endMonth));
    }

    @GetMapping("/subject/{subjectCode}/balance")
    public ResponseEntity<Map<String, Object>> getSubjectBalance(
            @PathVariable String subjectCode,
            @RequestParam Integer year,
            @RequestParam Integer month) {
        return ResponseEntity.ok(queryService.getSubjectBalance(subjectCode, year, month));
    }

    @GetMapping("/subject/{subjectCode}/entries")
    public ResponseEntity<List<GeneralLedger.GeneralLedgerEntry>> getEntriesBySubjectCode(
            @PathVariable String subjectCode) {
        return ResponseEntity.ok(queryService.getEntriesBySubjectCode(subjectCode));
    }
}