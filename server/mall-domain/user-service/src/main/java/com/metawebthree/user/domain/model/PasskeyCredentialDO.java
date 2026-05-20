package com.metawebthree.user.domain.model;

import com.baomidou.mybatisplus.annotation.TableId;
import com.github.yulichang.annotation.Table;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Date;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Table("PasskeyCredential")
public class PasskeyCredentialDO {
    @TableId
    private Long id;
    private Long userId;
    private String credentialId;
    private String publicKey;
    private String rpId;
    private Long counter;
    private String deviceType;
    private Date createdAt;
    private Date lastUsedAt;
    private Boolean isRevoked;
}
