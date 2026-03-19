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
@Table("TokenMapping")
public class TokenMappingDO {
    @TableId
    private Long id;
    private String parentToken;
    private String childToken;
    private Long userId;
    private String permissions;
    private Date expiresAt;
    private Boolean isRevoked;
    private Date createdAt;
    private Date updatedAt;
}