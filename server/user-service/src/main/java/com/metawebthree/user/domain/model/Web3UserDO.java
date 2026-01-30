package com.metawebthree.user.domain.model;

import com.baomidou.mybatisplus.annotation.TableName;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@TableName("Web3_User")
public class Web3UserDO {
    private Long id;
    private Long userId;
    private String walletAddress;
}
