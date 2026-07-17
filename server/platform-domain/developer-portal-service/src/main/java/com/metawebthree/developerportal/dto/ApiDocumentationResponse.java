package com.metawebthree.developerportal.dto;

import lombok.Data;
import java.util.List;

@Data
public class ApiDocumentationResponse {

    private String openapi = "3.0.0";
    private Info info;
    private List<Server> servers;
    private Paths paths;
    private Components components;

    @Data
    public static class Info {
        private String title;
        private String description;
        private String version;
        private Contact contact;
    }

    @Data
    public static class Contact {
        private String name;
        private String email;
        private String url;
    }

    @Data
    public static class Server {
        private String url;
        private String description;
    }

    @Data
    public static class Paths {
    }

    @Data
    public static class Components {
        private SecuritySchemes securitySchemes;
        private Schemas schemas;
    }

    @Data
    public static class SecuritySchemes {
        private ApiKeyAuth apiKeyAuth;
        private OAuth2 oauth2;
    }

    @Data
    public static class ApiKeyAuth {
        private String type = "apiKey";
        private String name = "X-API-Key";
        private String in = "header";
    }

    @Data
    public static class OAuth2 {
        private String type = "oauth2";
        private OAuthFlows flows;
    }

    @Data
    public static class OAuthFlows {
        private AuthorizationCode authorizationCode;
        private ClientCredentials clientCredentials;
    }

    @Data
    public static class AuthorizationCode {
        private String authorizationUrl;
        private String tokenUrl;
        private String refreshUrl;
        private List<String> scopes;
    }

    @Data
    public static class ClientCredentials {
        private String tokenUrl;
        private List<String> scopes;
    }

    @Data
    public static class Schemas {
    }
}
