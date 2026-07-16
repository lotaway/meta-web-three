package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.metawebthree.forecasting.domain.entity.ForecastModel;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public interface ForecastModelJpaRepository extends JpaRepository<ForecastModel, Long> {

    Optional<ForecastModel> findByModelName(String modelName);

    Optional<ForecastModel> findByModelTypeAndStatus(
        String modelType, ForecastModel.ModelStatus status);

    List<ForecastModel> findByStatus(ForecastModel.ModelStatus status);
}
