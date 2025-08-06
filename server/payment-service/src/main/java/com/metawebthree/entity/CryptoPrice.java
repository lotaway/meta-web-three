package com.metawebthree.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "crypto_prices")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CryptoPrice {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private String symbol; // BTC-USD, ETH-USD, etc.
    
    @Column(nullable = false)
    private String baseCurrency; // BTC, ETH, USDT
    
    @Column(nullable = false)
    private String quoteCurrency; // USD, CNY, EUR
    
    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal price;
    
    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal bidPrice;
    
    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal askPrice;
    
    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal volume24h;
    
    @Column(nullable = false, precision = 20, scale = 8)
    private BigDecimal change24h;
    
    @Column(nullable = false, precision = 5, scale = 2)
    private BigDecimal changePercent24h;
    
    @Column(nullable = false)
    private String source; // binance, coinbase, okx
    
    @Column(nullable = false)
    private LocalDateTime timestamp;
    
    @Column(nullable = false)
    private LocalDateTime createdAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
} 