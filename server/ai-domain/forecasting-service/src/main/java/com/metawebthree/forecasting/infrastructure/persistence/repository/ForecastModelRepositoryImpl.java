package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.metawebthree.forecasting.domain.entity.ForecastModel;
import com.metawebthree.forecasting.domain.repository.ForecastModelRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class ForecastModelRepositoryImpl implements ForecastModelRepository {

    private final Map<Long, ForecastModel> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);

    @Override
    public Optional<ForecastModel> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }

    @Override
    public Optional<ForecastModel> findByModelName(String modelName) {
        return storage.values().stream()
            .filter(m -> m.getModelName().equals(modelName))
            .findFirst();
    }

    @Override
    public Optional<ForecastModel> findByModelTypeAndStatus(
            String modelType, ForecastModel.ModelStatus status) {
        return storage.values().stream()
            .filter(m -> m.getModelType().equals(modelType))
            .filter(m -> m.getStatus() == status)
            .findFirst();
    }

    @Override
    public List<ForecastModel> findByStatus(ForecastModel.ModelStatus status) {
        return storage.values().stream()
            .filter(m -> m.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public List<ForecastModel> findAll() {
        return new ArrayList<>(storage.values());
    }

    @Override
    public ForecastModel save(ForecastModel model) {
        if (model.getId() == null) {
            model.setId(idGenerator.getAndIncrement());
        }
        storage.put(model.getId(), model);
        return model;
    }

    @Override
    public void update(ForecastModel model) {
        if (model.getId() != null && storage.containsKey(model.getId())) {
            storage.put(model.getId(), model);
        }
    }

    @Override
    public void deleteById(Long id) {
        storage.remove(id);
    }
}