package com.metawebthree.service.impl;

import lombok.RequiredArgsConstructor;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Repository;

import com.metawebthree.service.ModelScorerService;

import java.io.IOException;
import java.util.Map;

@Repository
@RequiredArgsConstructor
public class ModelScorerServiceImpl implements ModelScorerService {

    @Value("${ai.model.service.url}")
    private String modelServiceUrl;

    private final OkHttpClient httpClient;

    @Override
    public int score(String scene, Map<String, Object> features) {
        RequestBody requestBody = RequestBody.create(
                String.format("{\"scene\":\"%s\",\"features\":%s}", scene, features),
                MediaType.parse("application/json"));

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
        double d = debt instanceof Number number ? number.doubleValue() : 0d;
        Object age = features.get("age");
        int a = age instanceof Number number ? number.intValue() : 30;
        double s = 700 - d * 120 - Math.max(0, 25 - a) * 2;
        return (int) Math.round(s);
    }
}
