package com.metawebthree.developerportal.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("oauth_application")
public class OAuthApplication {

    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("client_id")
    private String clientId;

    @TableField("client_secret")
    private String clientSecret;

    @TableField("developer_id")
    private String developerId;

    @TableField("name")
    private String name;

    @TableField("description")
    private String description;

    @TableField("redirect_uris")
    private String redirectUris;

    @TableField("app_type")
    private AppType appType = AppType.CONFIDENTIAL;

    @TableField("grant_types")
    private String grantTypes = "authorization_code,refresh_token";

    @TableField("scopes")
    private String scopes;

    @TableField("status")
    private AppStatus status = AppStatus.ACTIVE;

    @TableField("created_at")
    private LocalDateTime createdAt;

    @TableField("updated_at")
    private LocalDateTime updatedAt;

    public enum AppType { CONFIDENTIAL, PUBLIC }

    public enum AppStatus { ACTIVE, DISABLED, SUSPENDED }
}
