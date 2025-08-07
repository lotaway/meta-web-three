package com.metawebthree.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.entity.CryptoPrice;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface CryptoPriceRepository extends BaseMapper<CryptoPrice> {
    
    @Select("SELECT * FROM crypto_prices WHERE symbol = #{symbol} ORDER BY timestamp DESC LIMIT 1")
    CryptoPrice findFirstBySymbolOrderByTimestampDesc(@Param("symbol") String symbol);
    
    @Select("SELECT * FROM crypto_prices WHERE base_currency = #{baseCurrency} AND quote_currency = #{quoteCurrency}")
    List<CryptoPrice> findByBaseCurrencyAndQuoteCurrency(@Param("baseCurrency") String baseCurrency, @Param("quoteCurrency") String quoteCurrency);
    
    @Select("SELECT * FROM crypto_prices WHERE symbol = #{symbol} AND timestamp >= #{startTime} ORDER BY timestamp DESC")
    List<CryptoPrice> findBySymbolAndTimestampAfter(@Param("symbol") String symbol, @Param("startTime") LocalDateTime startTime);
    
    @Select("SELECT * FROM crypto_prices WHERE base_currency = #{baseCurrency} AND quote_currency = #{quoteCurrency} AND source = #{source} ORDER BY timestamp DESC LIMIT 1")
    CryptoPrice findLatestBySymbolAndSource(@Param("baseCurrency") String baseCurrency, 
                                           @Param("quoteCurrency") String quoteCurrency, 
                                           @Param("source") String source);
    
    @Select("SELECT * FROM crypto_prices WHERE base_currency = #{baseCurrency} AND quote_currency = #{quoteCurrency} ORDER BY timestamp DESC LIMIT 1")
    CryptoPrice findLatestBySymbol(@Param("baseCurrency") String baseCurrency, 
                                   @Param("quoteCurrency") String quoteCurrency);
    
    @Select("SELECT * FROM crypto_prices WHERE source = #{source}")
    List<CryptoPrice> findBySource(@Param("source") String source);
    
    @Select("SELECT DISTINCT symbol FROM crypto_prices")
    List<String> findAllSymbols();
} 