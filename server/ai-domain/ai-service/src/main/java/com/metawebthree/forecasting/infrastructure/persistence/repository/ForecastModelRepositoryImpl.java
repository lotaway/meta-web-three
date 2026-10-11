package com.metawebthree.forecasting.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.metawebthree.forecasting.domain.entity.ForecastModel;
import com.metawebthree.forecasting.domain.repository.ForecastModelRepository;
import com.metawebthree.forecasting.infrastructure.persistence.mapper.ForecastModelMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public class ForecastModelRepositoryImpl implements ForecastModelRepository {

    private final ForecastModelMapper mapper;

    public ForecastModelRepositoryImpl(ForecastModelMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<ForecastModel> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public Optional<ForecastModel> findByModelName(String modelName) {
        return Optional.ofNullable(
            mapper.selectOne(new QueryWrapper<ForecastModel>().eq("model_name", modelName)));
    }

    @Override
    public Optional<ForecastModel> findByModelTypeAndStatus(
            String modelType, ForecastModel.ModelStatus status) {
        return Optional.ofNullable(
            mapper.selectOne(new QueryWrapper<ForecastModel>()
                .eq("model_type", modelType).eq("status", status)));
    }

    @Override
    public List<ForecastModel> findByStatus(ForecastModel.ModelStatus status) {
        return mapper.selectList(new QueryWrapper<ForecastModel>().eq("status", status));
    }

    @Override
    public List<ForecastModel> findAll() {
        return mapper.selectList(null);
    }

    @Override
    public void save(ForecastModel model) {
        mapper.insert(model);
    }

    @Override
    public void update(ForecastModel model) {
        mapper.updateById(model);
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
