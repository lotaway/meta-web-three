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
@TableName("tb_community_member")
public class CommunityMemberDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private Long communityId;
    private Long userId;
    private String role;
    private String nickname;
    private Integer messageCount;
    private String status;
    private Timestamp joinedAt;
    private Timestamp lastActiveAt;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}