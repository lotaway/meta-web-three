package com.metawebthree.recommendation.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.recommendation.domain.aishopping.entity.AiSearchLog;
import com.metawebthree.recommendation.domain.aishopping.repository.AiSearchLogRepository;
import com.metawebthree.recommendation.infrastructure.persistence.entity.AiSearchLogDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.AiSearchLogMapper;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.stereotype.Repository;

@Repository
public class AiSearchLogRepositoryImpl implements AiSearchLogRepository {

    private final AiSearchLogMapper logMapper;

    public AiSearchLogRepositoryImpl(AiSearchLogMapper logMapper) {
        this.logMapper = logMapper;
    }

    @Override
    public AiSearchLog save(AiSearchLog log) {
        AiSearchLogDO entity = toDO(log);
        if (log.getId() == null) {
            logMapper.insert(entity);
            log.setId(entity.getId());
        } else {
            logMapper.updateById(entity);
        }
        return log;
    }

    @Override
    public List<AiSearchLog> findRecent(int limit) {
        LambdaQueryWrapper<AiSearchLogDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.orderByDesc(AiSearchLogDO::getCreatedAt).last("LIMIT " + limit);
        return logMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }

    @Override
    public long count() {
        return logMapper.selectCount(new LambdaQueryWrapper<>());
    }

    private AiSearchLog toDomain(AiSearchLogDO entity) {
        AiSearchLog log = new AiSearchLog();
        log.setId(entity.getId());
        log.setUserId(entity.getUserId());
        log.setSearchType(entity.getSearchType() != null
                ? AiSearchLog.SearchType.valueOf(entity.getSearchType()) : null);
        log.setQueryText(entity.getQueryText());
        log.setCorrectedText(entity.getCorrectedText());
        log.setResultCount(entity.getResultCount());
        log.setResponseTimeMs(entity.getResponseTimeMs());
        log.setCreatedAt(entity.getCreatedAt());
        return log;
    }

    private AiSearchLogDO toDO(AiSearchLog log) {
        AiSearchLogDO entity = new AiSearchLogDO();
        entity.setId(log.getId());
        entity.setUserId(log.getUserId());
        entity.setSearchType(log.getSearchType() != null ? log.getSearchType().name() : null);
        entity.setQueryText(log.getQueryText());
        entity.setCorrectedText(log.getCorrectedText());
        entity.setResultCount(log.getResultCount());
        entity.setResponseTimeMs(log.getResponseTimeMs());
        entity.setCreatedAt(log.getCreatedAt() != null ? log.getCreatedAt() : LocalDateTime.now());
        return entity;
    }
}
