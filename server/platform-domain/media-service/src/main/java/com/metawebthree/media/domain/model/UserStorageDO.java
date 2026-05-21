package com.metawebthree.media.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("tb_user_storage")
public class UserStorageDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long userId;
    private Long totalUsed;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
