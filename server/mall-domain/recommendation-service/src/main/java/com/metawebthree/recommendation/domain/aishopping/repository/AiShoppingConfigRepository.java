package com.metawebthree.recommendation.domain.aishopping.repository;

import com.metawebthree.recommendation.domain.aishopping.entity.AiShoppingConfig;
import java.util.List;
import java.util.Optional;

public interface AiShoppingConfigRepository {

    Optional<String> findValue(String configKey);

    Optional<AiShoppingConfig> findById(String configKey);

    AiShoppingConfig save(AiShoppingConfig config);

    List<AiShoppingConfig> findAll();

    void delete(String configKey);
}
