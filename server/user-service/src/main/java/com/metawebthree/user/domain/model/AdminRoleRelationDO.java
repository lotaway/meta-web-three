package com.metawebthree.user.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_admin_role_relation")
public class AdminRoleRelationDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long adminId;
    private Long roleId;
}
