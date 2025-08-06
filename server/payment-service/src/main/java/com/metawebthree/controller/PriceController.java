package com.metawebthree.controller;

import com.metawebthree.dto.CryptoPriceResponse;
import com.metawebthree.service.PriceEngineService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.List;

@RestController
@RequestMapping("/api/v1/prices")
@RequiredArgsConstructor
@Slf4j
public class PriceController {
    
    private final PriceEngineService priceEngineService;
    
    /**
     * 获取实时价格
     */
    @GetMapping("/{symbol}")
    public ResponseEntity<CryptoPriceResponse> getCurrentPrice(@PathVariable String symbol) {
        log.info("Getting current price for symbol: {}", symbol);
        
        var price = priceEngineService.getCurrentPrice(symbol);
        
        CryptoPriceResponse response = CryptoPriceResponse.builder()
                .symbol(price.getSymbol())
                .baseCurrency(price.getBaseCurrency())
                .quoteCurrency(price.getQuoteCurrency())
                .price(price.getPrice())
                .bidPrice(price.getBidPrice())
                .askPrice(price.getAskPrice())
                .volume24h(price.getVolume24h())
                .change24h(price.getChange24h())
                .changePercent24h(price.getChangePercent24h())
                .source(price.getSource())
                .timestamp(price.getTimestamp())
                .updatedAt(price.getCreatedAt())
                .build();
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * 获取加权平均价格
     */
    @GetMapping("/weighted/{baseCurrency}/{quoteCurrency}")
    public ResponseEntity<BigDecimal> getWeightedAveragePrice(
            @PathVariable String baseCurrency,
            @PathVariable String quoteCurrency) {
        
        log.info("Getting weighted average price for {}-{}", baseCurrency, quoteCurrency);
        
        BigDecimal price = priceEngineService.getWeightedAveragePrice(baseCurrency, quoteCurrency);
        
        return ResponseEntity.ok(price);
    }
    
    /**
     * 计算兑换汇率
     */
    @GetMapping("/exchange-rate")
    public ResponseEntity<BigDecimal> calculateExchangeRate(
            @RequestParam String fromCurrency,
            @RequestParam String toCurrency,
            @RequestParam BigDecimal amount) {
        
        log.info("Calculating exchange rate: {} {} to {}", amount, fromCurrency, toCurrency);
        
        BigDecimal rate = priceEngineService.calculateExchangeRate(fromCurrency, toCurrency, amount);
        
        return ResponseEntity.ok(rate);
    }
    
    /**
     * 获取价格变化百分比
     */
    @GetMapping("/{symbol}/change")
    public ResponseEntity<BigDecimal> getPriceChange(
            @PathVariable String symbol,
            @RequestParam(defaultValue = "24") int hours) {
        
        log.info("Getting price change for {} over {} hours", symbol, hours);
        
        BigDecimal change = priceEngineService.getPriceChangePercent(symbol, hours);
        
        return ResponseEntity.ok(change);
    }
} 