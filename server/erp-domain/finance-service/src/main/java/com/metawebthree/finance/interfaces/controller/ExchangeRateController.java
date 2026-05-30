package com.metawebthree.finance.interfaces.controller;

import com.metawebthree.finance.application.command.exchange.ExchangeRateCommandService;
import com.metawebthree.finance.application.query.exchange.ExchangeRateQueryService;
import com.metawebthree.finance.domain.entity.exchange.ExchangeRate;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/exchange-rates")
@RequiredArgsConstructor
public class ExchangeRateController {

    private final ExchangeRateQueryService exchangeRateQueryService;
    private final ExchangeRateCommandService exchangeRateCommandService;

    @GetMapping
    public ResponseEntity<List<ExchangeRate>> getActiveRates() {
        return ResponseEntity.ok(exchangeRateQueryService.getActiveRates());
    }

    @GetMapping("/{id}")
    public ResponseEntity<ExchangeRate> getById(@PathVariable Long id) {
        ExchangeRate rate = exchangeRateQueryService.getById(id);
        if (rate == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(rate);
    }

    @GetMapping("/pair")
    public ResponseEntity<List<ExchangeRate>> getRatesByCurrencyPair(
            @RequestParam String sourceCurrency,
            @RequestParam String targetCurrency) {
        return ResponseEntity.ok(
                exchangeRateQueryService.getRatesByCurrencyPair(sourceCurrency, targetCurrency));
    }

    @GetMapping("/effective")
    public ResponseEntity<ExchangeRate> getEffectiveRate(
            @RequestParam String sourceCurrency,
            @RequestParam String targetCurrency,
            @RequestParam LocalDate date) {
        ExchangeRate rate = exchangeRateQueryService.getEffectiveRate(
                sourceCurrency, targetCurrency, date);
        if (rate == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(rate);
    }

    @PostMapping
    public ResponseEntity<ExchangeRate> createRate(@RequestBody ExchangeRateRequest request) {
        ExchangeRate rate = exchangeRateCommandService.createRate(
                request.sourceCurrency(),
                request.targetCurrency(),
                request.rate(),
                request.effectiveDate(),
                ExchangeRate.ExchangeRateType.valueOf(request.rateType()),
                request.createdBy());
        return ResponseEntity.ok(rate);
    }

    @PutMapping("/{id}")
    public ResponseEntity<ExchangeRate> updateRate(
            @PathVariable Long id,
            @RequestBody Map<String, BigDecimal> request) {
        BigDecimal newRate = request.get("rate");
        ExchangeRate rate = exchangeRateCommandService.updateRate(id, newRate);
        return ResponseEntity.ok(rate);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteRate(@PathVariable Long id) {
        exchangeRateCommandService.deleteRate(id);
        return ResponseEntity.noContent().build();
    }

    public record ExchangeRateRequest(
            String sourceCurrency,
            String targetCurrency,
            BigDecimal rate,
            LocalDate effectiveDate,
            String rateType,
            String createdBy) {
    }
}