package com.metawebthree.forecasting.domain.repository;

import com.metawebthree.forecasting.domain.entity.ForecastModel;
import java.util.List;
import java.util.Optional;

public interface ForecastModelRepository {
    Optional<ForecastModel> findById(Long id);
    Optional<ForecastModel> findByModelName(String modelName);
    Optional<ForecastModel> findByModelTypeAndStatus(
        String modelType, ForecastModel.ModelStatus status);
    List<ForecastModel> findByStatus(ForecastModel.ModelStatus status);
    List<ForecastModel> findAll();
    void save(ForecastModel model);
    void update(ForecastModel model);
    void deleteById(Long id);
}