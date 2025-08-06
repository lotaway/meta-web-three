package com.metawebthree.repository;

import com.metawebthree.entity.CryptoPrice;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface CryptoPriceRepository extends JpaRepository<CryptoPrice, Long> {
    
    Optional<CryptoPrice> findFirstBySymbolOrderByTimestampDesc(String symbol);
    
    List<CryptoPrice> findByBaseCurrencyAndQuoteCurrency(String baseCurrency, String quoteCurrency);
    
    @Query("SELECT cp FROM CryptoPrice cp WHERE cp.symbol = :symbol AND cp.timestamp >= :startTime ORDER BY cp.timestamp DESC")
    List<CryptoPrice> findBySymbolAndTimestampAfter(@Param("symbol") String symbol, @Param("startTime") java.time.LocalDateTime startTime);
    
    @Query("SELECT cp FROM CryptoPrice cp WHERE cp.baseCurrency = :baseCurrency AND cp.quoteCurrency = :quoteCurrency AND cp.source = :source ORDER BY cp.timestamp DESC LIMIT 1")
    Optional<CryptoPrice> findLatestBySymbolAndSource(@Param("baseCurrency") String baseCurrency, 
                                                      @Param("quoteCurrency") String quoteCurrency, 
                                                      @Param("source") String source);
    
    @Query("SELECT cp FROM CryptoPrice cp WHERE cp.baseCurrency = :baseCurrency AND cp.quoteCurrency = :quoteCurrency ORDER BY cp.timestamp DESC LIMIT 1")
    Optional<CryptoPrice> findLatestBySymbol(@Param("baseCurrency") String baseCurrency, 
                                             @Param("quoteCurrency") String quoteCurrency);
    
    List<CryptoPrice> findBySource(String source);
    
    @Query("SELECT DISTINCT cp.symbol FROM CryptoPrice cp")
    List<String> findAllSymbols();
} 