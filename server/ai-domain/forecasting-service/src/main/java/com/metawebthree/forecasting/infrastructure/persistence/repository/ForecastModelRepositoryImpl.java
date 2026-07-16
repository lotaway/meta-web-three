package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.metawebthree.forecasting.domain.entity.ForecastModel;
import com.metawebthree.forecasting.domain.repository.ForecastModelRepository;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public class ForecastModelRepositoryImpl implements ForecastModelRepository {

    private final ForecastModelJpaRepository jpaRepository;

    public ForecastModelRepositoryImpl(ForecastModelJpaRepository jpaRepository) {
        this.jpaRepository = jpaRepository;
    }

    @Override
    public Optional<ForecastModel> findById(Long id) {
        return jpaRepository.findById(id);
    }

    @Override
    public Optional<ForecastModel> findByModelName(String modelName) {
        return jpaRepository.findByModelName(modelName);
    }

    @Override
    public Optional<ForecastModel> findByModelTypeAndStatus(
            String modelType, ForecastModel.ModelStatus status) {
        return jpaRepository.findByModelTypeAndStatus(modelType, status);
    }

    @Override
    public List<ForecastModel> findByStatus(ForecastModel.ModelStatus status) {
        return jpaRepository.findByStatus(status);
    }

    @Override
    public List<ForecastModel> findAll() {
        return jpaRepository.findAll();
    }

    @Override
    public ForecastModel save(ForecastModel model) {
        return jpaRepository.save(model);
    }

    @Override
    public void update(ForecastModel model) {
        jpaRepository.save(model);
    }

    @Override
    public void deleteById(Long id) {
        jpaRepository.deleteById(id);
    }
}
