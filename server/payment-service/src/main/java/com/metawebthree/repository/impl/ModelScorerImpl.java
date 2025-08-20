package com.metawebthree.repository.impl;

import com.metawebthree.config.HttpClientConfig;
import com.metawebthree.repository.ModelScorer;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Repository;

import java.io.IOException;
import java.util.Map;

@Repository
public class ModelScorerImpl implements ModelScorer {

    private final HttpClientConfig httpClientConfig;

    @Value("${ai.model.service.url}")
    private String modelServiceUrl;

    private final OkHttpClient httpClient = new OkHttpClient();

    ModelScorerImpl(HttpClientConfig httpClientConfig) {
        this.httpClientConfig = httpClientConfig;
    }

    @Override
    public int score(String scene, Map<String, Object> features) {
        RequestBody requestBody = RequestBody.create(
            MediaType.parse("application/json"),
            String.format("{\"scene\":\"%s\",\"features\":%s}", scene, features)
        );

        Request request = new Request.Builder()
            .url(modelServiceUrl + "/score")
            .post(requestBody)
            .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                return fallbackScore(features);
            }
            
            String responseBody = response.body().string();
            return Integer.parseInt(responseBody);
        } catch (IOException e) {
            return fallbackScore(features);
        }
    }

    private int fallbackScore(Map<String, Object> features) {
        Object debt = features.get("external_debt_ratio");
        double d = debt instanceof Number ? ((Number)debt).doubleValue() : 0d;
        Object age = features.get("age");
        int a = age instanceof Number ? ((Number)age).intValue() : 30;
        double s = 700 - d * 120 - Math.max(0, 25 - a) * 2;
        return (int)Math.round(s);
    }
}
