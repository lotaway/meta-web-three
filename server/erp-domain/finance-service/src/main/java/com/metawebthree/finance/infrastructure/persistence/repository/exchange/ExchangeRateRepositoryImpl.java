package com.metawebthree.finance.infrastructure.persistence.repository.exchange;

import com.metawebthree.finance.domain.entity.exchange.ExchangeRate;
import com.metawebthree.finance.domain.repository.exchange.ExchangeRateRepository;
import com.metawebthree.finance.infrastructure.persistence.converter.exchange.ExchangeRateConverter;
import com.metawebthree.finance.infrastructure.persistence.dataobject.exchange.ExchangeRateDO;
import com.metawebthree.finance.infrastructure.persistence.mapper.exchange.ExchangeRateMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class ExchangeRateRepositoryImpl implements ExchangeRateRepository {

    private final ExchangeRateMapper exchangeRateMapper;
    private final ExchangeRateConverter exchangeRateConverter;

    @Override
    public Optional<ExchangeRate> findById(Long id) {
        ExchangeRateDO doObj = exchangeRateMapper.selectById(id);
        return Optional.ofNullable(exchangeRateConverter.toEntity(doObj));
    }

    @Override
    public List<ExchangeRate> findBySourceAndTargetCurrency(String sourceCurrency, String targetCurrency) {
        List<ExchangeRateDO> doObjs = exchangeRateMapper.selectBySourceAndTargetCurrency(sourceCurrency, targetCurrency);
        return doObjs.stream()
                .map(exchangeRateConverter::toEntity)
                .toList();
    }

    @Override
    public Optional<ExchangeRate> findEffectiveRate(String sourceCurrency, String targetCurrency, LocalDate date) {
        ExchangeRateDO doObj = exchangeRateMapper.selectEffectiveRate(sourceCurrency, targetCurrency, date);
        return Optional.ofNullable(exchangeRateConverter.toEntity(doObj));
    }

    @Override
    public List<ExchangeRate> findActiveRates() {
        List<ExchangeRateDO> doObjs = exchangeRateMapper.selectActiveRates();
        return doObjs.stream()
                .map(exchangeRateConverter::toEntity)
                .toList();
    }

    @Override
    public ExchangeRate save(ExchangeRate exchangeRate) {
        ExchangeRateDO doObj = exchangeRateConverter.toDo(exchangeRate);
        if (exchangeRate.getId() == null) {
            exchangeRateMapper.insert(doObj);
        } else {
            exchangeRateMapper.update(doObj);
        }
        return exchangeRate;
    }

    @Override
    public void delete(Long id) {
        exchangeRateMapper.deleteById(id);
    }
}