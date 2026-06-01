package com.metawebthree.socialcommerce.domain.model;

import java.sql.Timestamp;

import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.IdType;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("tb_community")
public class CommunityDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private String communityName;
    private String description;
    private Long ownerId;
    private String avatarUrl;
    private Integer memberCount;
    private Integer maxMembers;
    private String status;
    private String inviteCode;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}