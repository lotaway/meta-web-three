package com.metawebthree.developerportal.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("api_key")
public class ApiKey {

    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("key_id")
    private String keyId;

    @TableField("key_secret")
    private String keySecret;

    @TableField("developer_id")
    private String developerId;

    @TableField("name")
    private String name;

    @TableField("status")
    private KeyStatus status = KeyStatus.ACTIVE;

    @TableField("expires_at")
    private LocalDateTime expiresAt;

    @TableField("scopes")
    private String scopes;

    @TableField("allowed_ips")
    private String allowedIps;

    @TableField("allowed_domains")
    private String allowedDomains;

    @TableField("rate_limit")
    private Integer rateLimit = 100;

    @TableField("last_used_at")
    private LocalDateTime lastUsedAt;

    @TableField("created_at")
    private LocalDateTime createdAt;

    @TableField("updated_at")
    private LocalDateTime updatedAt;

    public enum KeyStatus {
        ACTIVE, DISABLED, EXPIRED, REVOKED
    }
}
