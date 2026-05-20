package com.metawebthree.forecasting.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class ForecastModel {
    private Long id;
    private String modelName;
    private String modelType;
    private String modelVersion;
    private ModelStatus status;
    private BigDecimal accuracy;
    private Integer trainingDays;
    private String featureConfig;
    private String algorithm;
    private LocalDateTime trainedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ModelStatus {
        DRAFT, TRAINING, TRAINED, DEPLOYED, DEPRECATED
    }

    public void create(String modelName, String modelType, String algorithm, 
                      String featureConfig, Integer trainingDays) {
        this.modelName = modelName;
        this.modelType = modelType;
        this.algorithm = algorithm;
        this.featureConfig = featureConfig;
        this.trainingDays = trainingDays;
        this.status = ModelStatus.DRAFT;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void startTraining() {
        if (status != ModelStatus.DRAFT) {
            throw new IllegalStateException("Can only start training from DRAFT status");
        }
        this.status = ModelStatus.TRAINING;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeTraining(BigDecimal accuracy) {
        if (status != ModelStatus.TRAINING) {
            throw new IllegalStateException("Model is not in TRAINING status");
        }
        this.accuracy = accuracy;
        this.trainedAt = LocalDateTime.now();
        this.status = ModelStatus.TRAINED;
        this.updatedAt = LocalDateTime.now();
    }

    public void deploy() {
        if (status != ModelStatus.TRAINED) {
            throw new IllegalStateException("Only TRAINED models can be deployed");
        }
        this.status = ModelStatus.DEPLOYED;
        this.updatedAt = LocalDateTime.now();
    }

    public void deprecate() {
        if (status != ModelStatus.DEPLOYED) {
            throw new IllegalStateException("Only DEPLOYED models can be deprecated");
        }
        this.status = ModelStatus.DEPRECATED;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isReadyForDeployment() {
        return status == ModelStatus.TRAINED && 
               accuracy != null && 
               accuracy.compareTo(BigDecimal.valueOf(70)) >= 0;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }
    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }
    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }
    public ModelStatus getStatus() { return status; }
    public void setStatus(ModelStatus status) { this.status = status; }
    public BigDecimal getAccuracy() { return accuracy; }
    public void setAccuracy(BigDecimal accuracy) { this.accuracy = accuracy; }
    public Integer getTrainingDays() { return trainingDays; }
    public void setTrainingDays(Integer trainingDays) { this.trainingDays = trainingDays; }
    public String getFeatureConfig() { return featureConfig; }
    public void setFeatureConfig(String featureConfig) { this.featureConfig = featureConfig; }
    public String getAlgorithm() { return algorithm; }
    public void setAlgorithm(String algorithm) { this.algorithm = algorithm; }
    public LocalDateTime getTrainedAt() { return trainedAt; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}