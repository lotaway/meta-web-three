package com.metawebthree.finance.infrastructure.persistence.mapper.exchange;

import com.metawebthree.finance.infrastructure.persistence.dataobject.exchange.ExchangeRateDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDate;
import java.util.List;

@Mapper
public interface ExchangeRateMapper {
    ExchangeRateDO selectById(@Param("id") Long id);

    List<ExchangeRateDO> selectBySourceAndTargetCurrency(
            @Param("sourceCurrency") String sourceCurrency,
            @Param("targetCurrency") String targetCurrency);

    ExchangeRateDO selectEffectiveRate(
            @Param("sourceCurrency") String sourceCurrency,
            @Param("targetCurrency") String targetCurrency,
            @Param("effectiveDate") LocalDate date);

    List<ExchangeRateDO> selectActiveRates();

    int insert(ExchangeRateDO exchangeRateDO);

    int update(ExchangeRateDO exchangeRateDO);

    int deleteById(@Param("id") Long id);
}