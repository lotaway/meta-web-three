package com.metawebthree.developerportal.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.*;
@Slf4j
@Service
@RequiredArgsConstructor
public class ApiDocumentationService {

    private final ApiDocumentationHelper documentationHelper;

    public Map<String, Object> generateOpenApiDocumentation(String baseUrl) {
        return documentationHelper.generateOpenApiDocumentation(baseUrl);
    }

    public Map<String, Object> generatePersonalizedDocumentation(String developerId, String baseUrl) {
        return documentationHelper.generatePersonalizedDocumentation(developerId, baseUrl);
    }

    public Map<String, String> generateSdkSamples(String language) {
        return documentationHelper.generateSdkSamples(language);
    }

    public Map<String, Object> generateSandboxTestData(String developerId) {
        return documentationHelper.generateSandboxTestData(developerId);
    }

    public Map<String, Object> resetSandboxEnvironment(String developerId) {
        return documentationHelper.resetSandboxEnvironment(developerId);
    }
}
