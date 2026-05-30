package com.metawebthree.finance.adapter.http;

import com.metawebthree.finance.application.query.FinancialRatioQueryService;
import com.metawebthree.finance.domain.entity.FinancialRatio;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/finance/financial-ratio")
public class FinancialRatioController {
    private final FinancialRatioQueryService queryService;

    public FinancialRatioController(FinancialRatioQueryService queryService) {
        this.queryService = queryService;
    }

    @GetMapping("/dashboard")
    public ResponseEntity<Map<String, Object>> getDashboard(
            @RequestParam(defaultValue = "MONTHLY") String period) {
        Map<String, Object> result = queryService.getDashboardRatios(period);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/details")
    public ResponseEntity<Map<String, Object>> getRatioDetails(
            @RequestParam String ratioType,
            @RequestParam(defaultValue = "") String period) {
        Map<String, Object> result = queryService.getRatioDetails(ratioType, period);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/comparison")
    public ResponseEntity<Map<String, Object>> comparePeriods(
            @RequestParam String period1,
            @RequestParam String period2) {
        Map<String, Object> result = queryService.getRatioComparison(period1, period2);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/list")
    public ResponseEntity<List<FinancialRatio>> listAll() {
        List<FinancialRatio> ratios = queryService.getAllRatios();
        return ResponseEntity.ok(ratios);
    }

    @GetMapping("/current")
    public ResponseEntity<Map<String, Object>> getCurrentRatios(
            @RequestParam(defaultValue = "MONTHLY") String period) {
        Map<String, java.math.BigDecimal> ratios = queryService.calculateCurrentRatios(period);
        return ResponseEntity.ok(Map.of(
                "ratios", ratios,
                "period", period,
                "status", "success"
        ));
    }
}