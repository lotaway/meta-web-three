package com.metawebthree.recommendation.domain.aishopping.repository;

import com.metawebthree.recommendation.domain.aishopping.entity.AiSearchLog;
import java.util.List;

public interface AiSearchLogRepository {

    AiSearchLog save(AiSearchLog log);

    List<AiSearchLog> findRecent(int limit);

    long count();
}
