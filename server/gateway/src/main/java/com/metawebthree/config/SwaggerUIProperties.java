package com.metawebthree.config;

import java.util.List;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix = "springdoc.swagger-ui")
public class SwaggerUIProperties {
    private List<UrlConfig> urlConfigs;

    public List<UrlConfig> getUrlConfigs() {
        return urlConfigs;
    }

    public void setUrlConfigs(List<UrlConfig> urlConfigs) {
        this.urlConfigs = urlConfigs;
    }

    public static class UrlConfig {
        private String name;
        private String url;

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getUrl() {
            return url;
        }

        public void setUrl(String url) {
            this.url = url;
        }
    }
}
